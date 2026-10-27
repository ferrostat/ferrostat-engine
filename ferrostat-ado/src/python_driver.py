# Helper run inside the external Python process used by the `python`
# command. The host talks to it over stdin/stdout with framed messages:
#   TAG <length>\n<utf-8 payload>
# Host -> helper: EXEC (run a request), RET / FAIL (answer to a CALL).
# Helper -> host: HELLO (startup info), OUT / ERR (printed text),
# CALL (data or macro access while code runs), DONE (request finished).
import sys, os, io, ast, codeop, builtins, types, traceback, fnmatch, importlib, textwrap, re

_US, _RS = '\x1f', '\x1e'
_SYSMISS = 8.98846567431158e+307
_rfd, _wfd = os.dup(0), os.dup(1)
try:
    _null = os.open(os.devnull, os.O_RDWR)
    os.dup2(_null, 0)
    # stray output written straight to file descriptor 1 must not corrupt the protocol
    os.dup2(2, 1)
except OSError:
    pass
_rin = os.fdopen(_rfd, 'rb')
_rout = os.fdopen(_wfd, 'wb')


def _send(tag, text):
    b = text.encode('utf-8', 'replace')
    _rout.write(('%s %d\n' % (tag, len(b))).encode('ascii') + b)
    _rout.flush()


def _recv():
    head = _rin.readline()
    if not head:
        os._exit(0)
    tag, n = head.decode('ascii').split()
    n = int(n)
    buf = b''
    while len(buf) < n:
        chunk = _rin.read(n - len(buf))
        if not chunk:
            os._exit(0)
        buf += chunk
    return tag, buf.decode('utf-8')


class _Stream(io.TextIOBase):
    def __init__(self, tag):
        self._tag = tag

    def writable(self):
        return True

    def write(self, s):
        if s:
            _send(self._tag, str(s))
        return len(s)

    def flush(self):
        pass

    @property
    def encoding(self):
        return 'utf-8'

    def isatty(self):
        return False


sys.stdout = _Stream('OUT')
sys.stderr = _Stream('ERR')
sys.stdin = io.StringIO('')


def _call(name, *args):
    _send('CALL', _US.join([name] + [str(a) for a in args]))
    tag, payload = _recv()
    if tag == 'RET':
        return payload
    raise SystemError(payload)


def _decode(cell, missingval):
    k, v = cell[:1], cell[1:]
    if k == 'i':
        return int(float(v))
    if k == 'n':
        return float(v)
    if k == 'm':
        return _SYSMISS if missingval is None else missingval
    return v


def _encode(v):
    if v is None:
        return 'm'
    if isinstance(v, str):
        return 's' + v
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise TypeError('value %r cannot be stored' % (v,))
    if f != f or f >= _SYSMISS:
        return 'm'
    return 'n' + repr(f)


# ---------------------------------------------------------------- sfi shim

class SFIToolkit:
    @staticmethod
    def _plain(s):
        return re.sub(r'\{(txt|text|res|result|err|error|com|input|bf|it|sf|hline|p|break|reset)[^}]*\}', '', str(s))

    @staticmethod
    def display(s, asis=False):
        sys.stdout.write(SFIToolkit._plain(s) if not asis else str(s))

    @staticmethod
    def displayln(s, asis=False):
        SFIToolkit.display(s, asis)
        sys.stdout.write('\n')

    @staticmethod
    def errprint(s):
        sys.stderr.write(SFIToolkit._plain(s))

    @staticmethod
    def errprintln(s):
        sys.stderr.write(SFIToolkit._plain(s) + '\n')

    @staticmethod
    def stata(cmd, echo=False):
        _call('command', cmd, '1' if echo else '0')

    @staticmethod
    def macroExpand(s):
        return _call('expand', s)

    @staticmethod
    def formatValue(value, fmt):
        return _call('format', repr(float(value)), fmt)

    @staticmethod
    def strToName(s, prefix=False):
        n = re.sub(r'[^A-Za-z0-9_]', '_', s)[:32]
        if not n or n[0].isdigit():
            n = ('_' + n)[:32]
        return n

    @staticmethod
    def isValidName(name):
        return re.match(r'^[A-Za-z_][A-Za-z0-9_]{0,31}$', name) is not None

    @staticmethod
    def isValidVariableName(name):
        return SFIToolkit.isValidName(name)

    @staticmethod
    def getRealOfString(s):
        try:
            return float(s)
        except ValueError:
            return _SYSMISS

    @staticmethod
    def pollnow():
        pass

    @staticmethod
    def pollstd():
        pass


class Macro:
    @staticmethod
    def getLocal(name):
        return _call('getlocal', name)

    @staticmethod
    def setLocal(name, value):
        _call('setlocal', name, value)

    @staticmethod
    def getGlobal(name):
        return _call('getglobal', name)

    @staticmethod
    def setGlobal(name, value, vtype='visible'):
        _call('setglobal', name, value)


class Scalar:
    @staticmethod
    def getValue(name):
        v = _call('getscalar', name)
        return None if v == '' else _decode(v, None)

    @staticmethod
    def setValue(name, value, vtype='visible'):
        _call('setscalar', name, repr(float(value)))

    @staticmethod
    def getString(name):
        return _call('getstrscalar', name)

    @staticmethod
    def setString(name, value, vtype='visible'):
        _call('setstrscalar', name, value)


class Matrix:
    @staticmethod
    def _load(name):
        f = _call('getmatrix', name).split(_US)
        r, c = int(f[0]), int(f[1])
        vals = [_decode(x, None) for x in f[2:]]
        return [vals[i * c:(i + 1) * c] for i in range(r)]

    @staticmethod
    def get(name, rows=None, cols=None, missingval=None):
        m = Matrix._load(name)
        if rows is not None:
            m = [m[i] for i in ([rows] if isinstance(rows, int) else rows)]
        if cols is not None:
            cl = [cols] if isinstance(cols, int) else list(cols)
            m = [[row[j] for j in cl] for row in m]
        if missingval is not None:
            m = [[missingval if x >= _SYSMISS else x for x in row] for row in m]
        return m

    @staticmethod
    def getAt(name, row, col):
        return Matrix._load(name)[row][col]

    @staticmethod
    def getRowTotal(name):
        return len(Matrix._load(name))

    @staticmethod
    def getColTotal(name):
        m = Matrix._load(name)
        return len(m[0]) if m else 0

    @staticmethod
    def getRowNames(name):
        return _call('matnames', name, 'row').split(_US)

    @staticmethod
    def getColNames(name):
        return _call('matnames', name, 'col').split(_US)

    @staticmethod
    def setRowNames(name, names):
        _call('setmatnames', name, 'row', *names)

    @staticmethod
    def setColNames(name, names):
        _call('setmatnames', name, 'col', *names)

    @staticmethod
    def _put(name, m):
        r = len(m)
        c = len(m[0]) if r else 0
        _call('setmatrix', name, r, c, *[_encode(x) for row in m for x in row])

    @staticmethod
    def create(name, nrows, ncols, initialValue, rowNames=None, colNames=None):
        Matrix._put(name, [[initialValue] * ncols for _ in range(nrows)])
        if rowNames:
            Matrix.setRowNames(name, rowNames)
        if colNames:
            Matrix.setColNames(name, colNames)

    @staticmethod
    def store(name, val):
        if not isinstance(val, (list, tuple)):
            val = [[val]]
        elif val and not isinstance(val[0], (list, tuple)):
            val = [list(val)]
        Matrix._put(name, val)

    @staticmethod
    def storeAt(name, row, col, val):
        m = Matrix._load(name)
        m[row][col] = val
        Matrix._put(name, m)


class Data:
    @staticmethod
    def getObsTotal():
        return int(_call('obstotal'))

    @staticmethod
    def getVarCount():
        return int(_call('varcount'))

    @staticmethod
    def _name(var):
        return Data.getVarName(var) if isinstance(var, int) else _call('varlist', var).split(_US)[0]

    @staticmethod
    def _names(var):
        if var is None:
            return [Data.getVarName(i) for i in range(Data.getVarCount())]
        if isinstance(var, int):
            return [Data.getVarName(var)]
        if isinstance(var, str):
            return [n for n in _call('varlist', var).split(_US) if n]
        out = []
        for v in var:
            out.extend(Data._names(v))
        return out

    @staticmethod
    def _obs(obs, n):
        if obs is None:
            return list(range(n))
        if isinstance(obs, int):
            obs = [obs]
        obs = list(obs)
        for o in obs:
            if not 0 <= o < n:
                raise IndexError('observation index %d out of range' % o)
        return obs

    @staticmethod
    def _cols(names, valuelabel, missingval):
        if not names:
            return []
        raw = _call('getcols', '1' if valuelabel else '0', *names)
        return [[_decode(c, missingval) for c in col.split(_US)] if col else [] for col in raw.split(_RS)]

    @staticmethod
    def _rows(var, obs, selectvar, valuelabel, missingval):
        names = Data._names(var)
        n = Data.getObsTotal()
        obs = Data._obs(obs, n)
        cols = Data._cols(names, valuelabel, missingval)
        if selectvar is not None and selectvar != -1:
            sel = Data._cols([Data._name(selectvar)], False, None)[0]
            obs = [o for o in obs if sel[o] != 0 and sel[o] < _SYSMISS]
        elif selectvar == -1:
            obs = [o for o in obs if all(not (isinstance(c[o], float) and c[o] >= _SYSMISS) for c in cols)]
        return names, obs, cols

    @staticmethod
    def get(var=None, obs=None, selectvar=None, valuelabel=False, missingval=None):
        names, obs, cols = Data._rows(var, obs, selectvar, valuelabel, missingval)
        return [[c[o] for c in cols] for o in obs]

    @staticmethod
    def getAsDict(var=None, obs=None, selectvar=None, valuelabel=False, missingval=None):
        names, obs, cols = Data._rows(var, obs, selectvar, valuelabel, missingval)
        return {nm: [c[o] for o in obs] for nm, c in zip(names, cols)}

    @staticmethod
    def getAt(var, obs):
        return Data.get(var, obs)[0][0]

    @staticmethod
    def store(var, obs, val, selectvar=None):
        names = Data._names(var)
        n = Data.getObsTotal()
        obs = Data._obs(obs, n)
        if selectvar is not None:
            sel = Data._cols([Data._name(selectvar)], False, None)[0]
            obs = [o for o in obs if sel[o] != 0 and sel[o] < _SYSMISS]
        if not isinstance(val, (list, tuple)):
            rows = [[val] * len(names) for _ in obs]
        else:
            rows = [list(r) if isinstance(r, (list, tuple)) else [r] for r in val]
        if len(rows) != len(obs) or any(len(r) != len(names) for r in rows):
            raise ValueError('dimensions of values do not match the variables and observations')
        olist = ','.join(str(o) for o in obs)
        for j, nm in enumerate(names):
            _call('setcol', nm, olist, *[_encode(r[j]) for r in rows])

    @staticmethod
    def storeAt(var, obs, val):
        Data.store(var, obs, val)

    @staticmethod
    def getVarName(index):
        return _call('varname', index)

    @staticmethod
    def getVarIndex(name):
        return int(_call('varindex', name))

    @staticmethod
    def getVarType(var):
        return _call('vartype', Data._name(var))

    @staticmethod
    def isVarTypeStr(var):
        return Data.getVarType(var).startswith('str')

    @staticmethod
    def isVarTypeString(var):
        return Data.isVarTypeStr(var)

    @staticmethod
    def getVarLabel(var):
        return _call('varlabel', Data._name(var))

    @staticmethod
    def setVarLabel(var, label):
        _call('setvarlabel', Data._name(var), label)

    @staticmethod
    def getVarFormat(var):
        return _call('varformat', Data._name(var))

    @staticmethod
    def setVarFormat(var, fmt):
        _call('setvarformat', Data._name(var), fmt)

    @staticmethod
    def addVarByte(name):
        _call('addvar', name, 'byte')

    @staticmethod
    def addVarInt(name):
        _call('addvar', name, 'int')

    @staticmethod
    def addVarLong(name):
        _call('addvar', name, 'long')

    @staticmethod
    def addVarFloat(name):
        _call('addvar', name, 'float')

    @staticmethod
    def addVarDouble(name):
        _call('addvar', name, 'double')

    @staticmethod
    def addVarStr(name, length=1):
        _call('addvar', name, 'str%d' % max(1, min(int(length), 2045)))

    @staticmethod
    def addVarStrL(name):
        _call('addvar', name, 'strL')

    @staticmethod
    def setObsTotal(nobs):
        _call('setobs', int(nobs))

    @staticmethod
    def addObs(n):
        Data.setObsTotal(Data.getObsTotal() + int(n))

    @staticmethod
    def dropVar(var):
        _call('dropvar', *Data._names(var))

    @staticmethod
    def keepVar(var):
        keep = set(Data._names(var))
        drop = [nm for nm in Data._names(None) if nm not in keep]
        if drop:
            _call('dropvar', *drop)

    @staticmethod
    def renameVar(oldname, newname):
        _call('renamevar', Data._name(oldname), newname)


_sfi = types.ModuleType('sfi')
for _c in (Data, Macro, Scalar, Matrix, SFIToolkit):
    setattr(_sfi, _c.__name__, _c)
_sfi.SFIError = SystemError
sys.modules['sfi'] = _sfi

# ---------------------------------------------------------------- execution

_main = types.ModuleType('__main__')
sys.modules['__main__'] = _main
_ns = _main.__dict__
_ns['__builtins__'] = builtins


def _print_exc():
    et, ev, tb = sys.exc_info()
    # hide the helper's own frames
    while tb is not None and tb.tb_frame.f_code.co_filename == '<ferrostat>':
        tb = tb.tb_next
    sys.stderr.write(''.join(traceback.format_exception(et, ev, tb)))


def _echo(lines):
    for k, l in enumerate(lines):
        sys.stdout.write(('>>> ' if k == 0 else '... ') + l + '\n')


def _run_console(code, echo):
    # line by line like an interactive session (used when the whole block does not parse)
    buf, ok = [], True
    for line in code.split('\n') + ['']:
        if echo and (line or buf):
            sys.stdout.write(('... ' if buf else '>>> ') + line + '\n')
        buf.append(line)
        try:
            c = codeop.compile_command('\n'.join(buf), '<stdin>', 'single')
        except (SyntaxError, OverflowError, ValueError):
            _print_exc()
            buf, ok = [], False
            continue
        if c is None:
            continue
        buf = []
        try:
            exec(c, _ns)
        except BaseException:
            _print_exc()
            ok = False
    return ok


def _run(stop, echo, code):
    code = textwrap.dedent(code)
    try:
        tree = ast.parse(code, '<stdin>', 'exec')
    except SyntaxError:
        if stop:
            _print_exc()
            return False
        return _run_console(code, echo)
    lines = code.split('\n')
    ok = True
    for node in tree.body:
        if echo:
            _echo(lines[node.lineno - 1:getattr(node, 'end_lineno', node.lineno)])
        try:
            exec(compile(ast.Interactive(body=[node]), '<stdin>', 'single'), _ns)
        except BaseException:
            _print_exc()
            ok = False
            if stop:
                return False
    return ok


def _script(path, glob, *args):
    sys.argv = [path] + list(args)
    if glob:
        ns = _ns
    else:
        ns = {'__name__': '__main__', '__builtins__': builtins}
    ns['__file__'] = path
    with open(path, 'rb') as f:
        src = f.read()
    try:
        exec(compile(src, path, 'exec'), ns)
    except SystemExit as e:
        if e.code not in (None, 0):
            _print_exc()
            return False
    except BaseException:
        _print_exc()
        return False
    return True


def _user_names():
    return sorted(k for k in _ns if not k.startswith('__'))


def _describe(*pats):
    names = [k for k in _user_names() if not pats or any(fnmatch.fnmatchcase(k, p) for p in pats)]
    if not names:
        return
    w = max(12, max(len(k) for k in names) + 2)
    print('%-*s%s' % (w, 'name', 'type'))
    print('-' * (w + 20))
    for k in names:
        print('%-*s%s' % (w, k, type(_ns[k]).__name__))


def _drop(*pats):
    for k in _user_names():
        if any(fnmatch.fnmatchcase(k, p) for p in pats):
            del _ns[k]


def _clear():
    for k in _user_names():
        del _ns[k]


def _which(name):
    m = importlib.import_module(name)
    print(repr(m))


def _userpath(prepend, *paths):
    for p in paths:
        if p in sys.path:
            sys.path.remove(p)
        if prepend:
            sys.path.insert(0, p)
        else:
            sys.path.append(p)


_admin = {'describe': _describe, 'drop': _drop, 'clear': _clear, 'which': _which, 'userpath': _userpath}

_send('HELLO', _US.join([sys.version.split()[0], sys.executable or '', '%d-bit' % (64 if sys.maxsize > 2 ** 32 else 32), os.path.dirname(os.__file__)]))

while True:
    tag, payload = _recv()
    if tag != 'EXEC':
        continue
    f = payload.split(_US)
    mode, cwd = f[0], f[1]
    try:
        if cwd and os.path.isdir(cwd):
            os.chdir(cwd)
    except OSError:
        pass
    try:
        if mode == 'run':
            ok = _run(f[2] == '1', f[3] == '1', _US.join(f[4:]))
        elif mode == 'script':
            ok = _script(f[2], f[3] == '1', *f[4:])
        else:
            try:
                _admin[f[2]](*[a == '1' if f[2] == 'userpath' and i == 0 else a for i, a in enumerate(f[3:])])
                ok = True
            except BaseException:
                _print_exc()
                ok = False
    except BaseException:
        _print_exc()
        ok = False
    try:
        sys.stdout.flush()
    except Exception:
        pass
    _send('DONE', '0' if ok else '1')
