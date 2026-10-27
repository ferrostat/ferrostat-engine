//! `python` integration through an external interpreter process. Tests are
//! skipped (pass trivially) when no Python 3 is installed.

use ferrostat_tests::*;

/// The name of a working Python 3 on PATH (`python3` on Unix, usually
/// only `python` on Windows).
fn python_name() -> Option<&'static str> {
    ["python3", "python"].into_iter().find(|p| {
        std::process::Command::new(p).arg("--version").output().is_ok_and(|o| o.status.success() && (String::from_utf8_lossy(&o.stdout).contains("Python 3") || String::from_utf8_lossy(&o.stderr).contains("Python 3")))
    })
}

fn have_python() -> bool {
    python_name().is_some()
}

macro_rules! need_python {
    () => {
        if !have_python() {
            eprintln!("python not found; skipped");
            return;
        }
    };
}

#[test]
fn one_liners_share_state() {
    need_python!();
    let mut s = Session::new();
    assert_eq!(s.run("python: print('Hello')").trim(), "Hello");
    s.run("local a = 2");
    s.run("python: b = 2*`a'");
    assert_eq!(s.run("python: print(b)").trim(), "4");
    // the next block still sees b
    let out = s.try_run("python:\nc = b // 3\nprint(c * 10)\nend").unwrap();
    assert!(out.contains("10"), "{out}");
}

#[test]
fn block_keeps_indentation_and_echoes_expressions() {
    need_python!();
    let mut s = Session::new();
    let out = s
        .try_run("python\na1 = 1\nb = a1 + 2\nif b % 2 == 0:\n\tprint('even')\nelse:\n\tprint('odd')\n\n1+1\nend")
        .unwrap();
    assert!(out.contains("odd"), "{out}");
    assert!(out.contains('2'), "{out}");
}

#[test]
fn errors_stop_colon_blocks_only() {
    need_python!();
    let mut s = Session::new();
    // without the colon every statement runs and errors are only shown
    let out = s.try_run("python\n1+'test'\nprint('after')\nend").unwrap();
    assert!(out.contains("TypeError") && out.contains("after"), "{out}");
    // with the colon the first error ends the block with r(7102)
    let e = s.try_run("python:\n1+'test'\nprint('after')\nend").unwrap_err();
    assert!(e.contains("r(7102)"), "{e}");
    let e = s.try_run("python: raise ValueError('boom')").unwrap_err();
    assert!(e.contains("r(7102)"), "{e}");
    assert!(s.try_run("capture python: import no_such_module_xyz").is_ok());
    assert_eq!(s.scalar("_rc"), 7102.0);
}

#[test]
fn sfi_data_access() {
    need_python!();
    let mut s = Session::with_fixture();
    let out = s.try_run("python:\nfrom sfi import Data\nprint(Data.getObsTotal(), Data.getVarCount(), Data.getVarName(0))\nprint(Data.get(var='id', obs=[0, 1]))\nprint(Data.get('name', 0))\nend").unwrap();
    assert!(out.contains("12 8 id"), "{out}");
    assert!(out.contains("[[1], [2]]"), "{out}");
    assert!(out.contains("[['"), "{out}");
    let out = s.run("python: print(Data.getAsDict('d', obs=range(2), valuelabel=True))");
    assert!(out.contains("'d': ["), "{out}");
    s.try_run("python:\nData.addVarDouble('w')\nData.store('w', None, [i * 1.5 for i in range(12)])\nData.addVarStr('lab', 3)\nData.store('lab', 2, 'longer text')\nData.store('id', 0, 100000)\nend").unwrap();
    assert_eq!(s.nums("w")[3], 4.5);
    assert_eq!(s.strs("lab")[2], "longer text");
    assert_eq!(s.nums("id")[0], 100000.0);
    s.run("python: Data.setObsTotal(14)");
    assert_eq!(s.n(), 14);
}

#[test]
fn sfi_macros_scalars_matrices() {
    need_python!();
    let mut s = Session::new();
    s.run("local x hello");
    s.run("global g world");
    s.try_run("python:\nfrom sfi import Macro, Scalar, Matrix\nMacro.setLocal('y', Macro.getLocal('x') + ' ' + Macro.getGlobal('g'))\nMacro.setGlobal('h', 'gg')\nScalar.setValue('r(rate)', 0.25)\nScalar.setValue('sc', 3)\nMatrix.store('M', [[1, 2], [3, 4]])\nend").unwrap();
    assert_eq!(s.run("display \"`y'\"").trim(), "hello world");
    assert_eq!(s.run("display \"$h\"").trim(), "gg");
    assert_eq!(s.scalar("r(rate)"), 0.25);
    assert_eq!(s.scalar("sc"), 3.0);
    assert_eq!(s.scalar("M[2,1]"), 3.0);
    assert_eq!(s.run("python: print(Scalar.getValue('sc'), Matrix.get('M')[1])").trim(), "3.0 [3.0, 4.0]");
}

#[test]
fn sfi_runs_commands() {
    need_python!();
    let mut s = Session::with_fixture();
    let out = s.try_run("python:\nfrom sfi import SFIToolkit\nSFIToolkit.stata('generate ten = 10')\nSFIToolkit.displayln('{res}done')\nend").unwrap();
    assert!(s.has_var("ten"));
    assert!(out.contains("done") && !out.contains("{res}"), "{out}");
    let e = s.try_run("python: SFIToolkit.stata('no_such_command_xyz')").unwrap_err();
    assert!(e.contains("r(7102)"), "{e}");
}

#[test]
fn workspace_commands_and_scripts() {
    need_python!();
    let mut s = Session::new();
    s.run("python: alpha = 1");
    s.run("python: beta = 'x'");
    let out = s.run("python describe");
    assert!(out.contains("alpha") && out.contains("beta"), "{out}");
    s.run("python drop a*");
    let out = s.run("python describe");
    assert!(!out.contains("alpha") && out.contains("beta"), "{out}");
    s.run("python clear");
    assert!(!s.run("python describe").contains("beta"));
    assert!(s.run("python query").contains("version"));
    assert!(s.run("python which os").contains("module 'os'"));
    s.fails("python which no_such_module_xyz");

    let dir = temp_dir("t13_python_script");
    let f = dir.join("hello.py");
    std::fs::write(&f, "import sys\nfrom sfi import Macro\nMacro.setLocal('got', ' '.join(sys.argv[1:]))\nprint('script ran')\n").unwrap();
    let out = s.run(&format!("python script \"{}\", args(one two)", f.display()));
    assert!(out.contains("script ran"), "{out}");
    assert_eq!(s.run("display \"`got'\"").trim(), "one two");
}

#[test]
fn chosen_executable_is_used() {
    need_python!();
    let mut s = Session::new();
    s.run("set python_exec no_such_python_binary_xyz");
    s.fails("python: print(1)");
    assert!(s.run("python query").contains("initialized          no"));
    s.run(&format!("python set exec {}", python_name().unwrap()));
    assert_eq!(s.run("python: print(7)").trim(), "7");
}

#[cfg(unix)]
#[test]
fn shell_commands_still_work() {
    let mut s = Session::new();
    assert_eq!(s.run("shell echo hi").trim(), "hi");
    assert_eq!(s.run("! echo there").trim(), "there");
}
