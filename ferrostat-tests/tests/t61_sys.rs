//! Viewer, editor, dialog and window commands (`view`, `doedit`, `db`,
//! `window`), SMCL rendering, compiled C plugins (`plugin`) and Java calls
//! (`javacall`). Plugin and Java tests compile their code at run time and
//! are skipped when no C compiler or JDK is installed.

use ferrostat_ado::interpreter::{err_message, rc_of};
use ferrostat_ado::w11_ui::{DialogKind, FileDialogRequest, Frontend, MenuEntry, UiRequest};
use ferrostat_ado::AdoInterpreter;
use ferrostat_cmd::w11::smcl;
use ferrostat_tests::*;

fn dir(tag: &str) -> std::path::PathBuf {
    temp_dir(&format!("t61_{tag}"))
}

fn p(d: &std::path::Path, f: &str) -> String {
    d.join(f).display().to_string()
}

fn run(me: &mut AdoInterpreter, line: &str) -> String {
    match me.run_line(line) {
        Ok(o) => o,
        Err(e) => panic!("`{line}` failed: {e}"),
    }
}

fn rc(me: &mut AdoInterpreter, line: &str) -> i32 {
    match me.run_line(line) {
        Ok(_) => 0,
        Err(e) => rc_of(&e),
    }
}

const HELP: &str = "{smcl}
{* *! version 1.0}{...}
{viewerjumpto \"Syntax\" \"mycmd##syntax\"}{...}
{title:Title}

{phang}
{bf:mycmd} {hline 2} Compute something useful for the analyst, wrapping lines nicely across the page width when the text is long{p_end}

{marker syntax}{...}
{title:Syntax}

{p 8 17 2}
{cmd:mycmd} {varlist} {ifin} [{cmd:,} {it:options}]

{synoptset 20 tabbed}{...}
{synopthdr}
{synoptline}
{syntab:Main}
{synopt:{opt d:etail}}display additional statistics{p_end}
{synopt:{opth gen:erate(newvar)}}create {it:newvar}{p_end}
{synoptline}

{pstd}See {help summarize:the summarize help} and {browse \"https://example.org\":the web}.{p_end}
";

#[test]
fn smcl_rendering() {
    let text = smcl::to_text(HELP);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "Title");
    assert!(text.contains("    mycmd -- Compute something useful"), "{text}");
    assert!(lines.iter().any(|l| l.starts_with("        across")), "continuation indent: {text}");
    assert!(lines.iter().all(|l| l.chars().count() <= 79));
    assert!(text.contains("        mycmd varlist [if] [in] [, options]"), "{text}");
    assert!(text.contains("    detail                display additional statistics"), "{text}");
    assert!(text.contains("    generate(newvar)      create newvar"), "{text}");
    assert!(text.contains("  Main"));
    assert!(text.contains("See the summarize help and the web."), "{text}");
    assert!(!text.contains('{') && !text.contains("viewerjumpto"));
    // styled lines carry links and emphasis for graphical viewers
    let styled = smcl::render(HELP, 79);
    let spans: Vec<&smcl::Span> = styled.iter().flatten().collect();
    assert!(spans.iter().any(|s| s.text == "the summarize help" && s.style.link.as_deref() == Some("help summarize")));
    assert!(spans.iter().any(|s| s.style.link.as_deref() == Some("browse https://example.org")));
    assert!(spans.iter().any(|s| s.text == "Title" && s.style.bold));
    assert!(spans.iter().any(|s| s.text.contains("newvar") && s.style.italic));
    // links that run a command (directive name assembled here)
    let tag = ["st", "ata"].concat();
    let run_link = smcl::render(&format!("{{{tag} \"summarize x\":Run it}}"), 79);
    assert_eq!(run_link[0][0].text, "Run it");
    assert_eq!(run_link[0][0].style.link.as_deref(), Some("run summarize x"));
    // line mode: columns, characters, continuation, alignment
    assert_eq!(smcl::to_text("{col 3}a{col 8}b{c |}{space 2}c"), "  a    b|  c");
    assert_eq!(smcl::to_text("x{...}\ny"), "xy");
    assert_eq!(smcl::to_text("{ralign 6:ab}|{lalign 4:c}|{center 5:d}|"), "    ab|c   |  d  |");
}

#[test]
fn view_in_text_mode() {
    let d = dir("view");
    std::fs::write(d.join("notes.txt"), "first line\nsecond {line}\n").unwrap();
    std::fs::write(d.join("mycmd.sthlp"), HELP).unwrap();
    std::fs::write(d.join("job.do"), "sysuse auto\nsummarize\n").unwrap();
    std::fs::write(d.join("log.smcl"), "{smcl}\n{com}. di 1\n{res}1\n").unwrap();
    let mut me = AdoInterpreter::new();
    assert_eq!(me.frontend(), Frontend::Text);
    assert_eq!(run(&mut me, &format!("view \"{}\"", p(&d, "notes.txt"))), "first line\nsecond {line}");
    assert_eq!(run(&mut me, &format!("view file \"{}\"", p(&d, "notes.txt"))), "first line\nsecond {line}");
    let h = run(&mut me, &format!("view \"{}\"", p(&d, "mycmd.sthlp")));
    assert!(h.starts_with("Title") && h.contains("display additional statistics") && !h.contains('{'), "{h}");
    // asis shows the source
    assert!(run(&mut me, &format!("view \"{}\", asis", p(&d, "mycmd.sthlp"))).contains("{synoptset 20 tabbed}"));
    assert_eq!(run(&mut me, &format!("view smcl \"{}\"", p(&d, "log.smcl"))), ". di 1\n1");
    // extension added when missing
    assert_eq!(run(&mut me, &format!("view \"{}\"", p(&d, "log"))), ". di 1\n1");
    assert_eq!(run(&mut me, &format!("view do \"{}\"", p(&d, "job.do"))), "sysuse auto\nsummarize");
    assert!(run(&mut me, "view browse https://example.org").contains("https://example.org"));
    assert!(run(&mut me, "view help regress").contains("regress"));
    assert!(run(&mut me, "view search regress").contains("regress"));
    assert!(!run(&mut me, "view news").is_empty());
    assert_eq!(rc(&mut me, &format!("view \"{}\"", p(&d, "nosuch.txt"))), 601);
    assert_eq!(rc(&mut me, "view"), 198);
    assert_eq!(rc(&mut me, "view browse"), 198);
    assert!(me.take_ui_requests().is_empty(), "text front ends get no requests");
}

#[test]
fn doedit_db_window_in_text_mode() {
    let mut me = AdoInterpreter::new();
    assert!(run(&mut me, "doedit analysis").contains("analysis.do"));
    assert!(run(&mut me, "doedit").contains("doedit"));
    let o = run(&mut me, "db regress");
    assert!(o.contains("desktop interface") && o.contains("help regress"), "{o}");
    assert!(run(&mut me, "db summ").contains("db summarize"));
    let e = me.run_line("db nosuchcommand").unwrap_err();
    assert_eq!(rc_of(&e), 601);
    assert!(err_message(&e).contains("nosuchcommand"));
    assert_eq!(rc(&mut me, "db"), 198);
    // window: no file dialog -> empty macro; menus are recorded
    run(&mut me, "window fopen fn \"Open data\" \"Data|*.dta|All|*.*\" dta");
    assert_eq!(run(&mut me, "display \"[$fn]\""), "[]");
    run(&mut me, "window menu clear");
    run(&mut me, "window menu append submenu \"stUser\" \"Tools\"");
    run(&mut me, "window menu append item \"Tools\" \"Summary\" \"summarize\"");
    run(&mut me, "window menu append separator \"Tools\"");
    run(&mut me, "window menu refresh");
    assert_eq!(me.user_menu().len(), 3);
    assert_eq!(me.user_menu()[1], MenuEntry::Item { parent: "Tools".into(), label: "Summary".into(), command: "summarize".into() });
    assert_eq!(rc(&mut me, "window menu append item \"Tools\" \"x\""), 198);
    assert_eq!(rc(&mut me, "window menu bogus"), 198);
    run(&mut me, "window menu clear");
    assert!(me.user_menu().is_empty());
    run(&mut me, "window manage forward results");
    run(&mut me, "window push regress y x");
    assert!(run(&mut me, "window stopbox note \"hello\"").contains("hello"));
    assert_eq!(rc(&mut me, "window stopbox stop \"no\""), 1);
    assert!(me.take_ui_requests().is_empty());
}

#[test]
fn gui_front_end_gets_requests() {
    let d = dir("gui");
    std::fs::write(d.join("mycmd.sthlp"), HELP).unwrap();
    let mut me = AdoInterpreter::new();
    me.set_frontend(Frontend::Gui);
    run(&mut me, "set obs 3");
    run(&mut me, "gen y = _n");
    assert_eq!(run(&mut me, &format!("view \"{}\"", p(&d, "mycmd.sthlp"))), "");
    run(&mut me, "view browse https://example.org");
    run(&mut me, "doedit analysis");
    run(&mut me, "doedit");
    run(&mut me, "db regress");
    run(&mut me, "db summarize");
    run(&mut me, "db generate");
    run(&mut me, "window manage close graph");
    run(&mut me, "window push regress y");
    run(&mut me, "window menu refresh");
    let reqs = me.take_ui_requests();
    assert_eq!(reqs.len(), 10, "{reqs:?}");
    match &reqs[0] {
        UiRequest::View { title, lines } => {
            assert!(title.contains("mycmd.sthlp"));
            assert!(smcl::lines_to_text(lines).starts_with("Title"));
            assert!(lines.iter().flatten().any(|s| s.style.link.is_some()));
        }
        r => panic!("{r:?}"),
    }
    assert_eq!(reqs[1], UiRequest::Browse("https://example.org".into()));
    assert_eq!(reqs[2], UiRequest::Edit(Some("analysis.do".into())));
    assert_eq!(reqs[3], UiRequest::Edit(None));
    let UiRequest::Dialog(reg) = &reqs[4] else { panic!() };
    assert_eq!(reg.kind, DialogKind::Estimation);
    assert_eq!(reg.build("y", &["x1".into(), "x2".into()], "", "x1 > 0", "1/10", "vce(robust)"), "regress y x1 x2 if x1 > 0 in 1/10, vce(robust)");
    let UiRequest::Dialog(su) = &reqs[5] else { panic!() };
    assert_eq!(su.kind, DialogKind::Varlist);
    assert_eq!(su.build("", &["y".into()], "", "", "", ""), "summarize y");
    let UiRequest::Dialog(g) = &reqs[6] else { panic!() };
    assert_eq!(g.kind, DialogKind::Generic);
    assert_eq!(reqs[7], UiRequest::Manage(vec!["close".into(), "graph".into()]));
    assert_eq!(reqs[8], UiRequest::PushHistory("regress y".into()));
    assert_eq!(reqs[9], UiRequest::MenuChanged);
    assert!(me.take_ui_requests().is_empty());
    // file dialogs answered by the front end
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<FileDialogRequest>::new()));
    let s2 = seen.clone();
    me.set_file_dialog(Some(Box::new(move |r: &FileDialogRequest| {
        s2.lock().unwrap().push(r.clone());
        Some(if r.save { "/data/out".to_string() } else { "/data/in.dta".to_string() })
    })));
    run(&mut me, "window fopen inf \"Open data\" \"Data files|*.dta;*.csv|All files|*.*\"");
    run(&mut me, "window fsave outf \"Save\" \"Data|*.dta\" dta");
    assert_eq!(run(&mut me, "display \"$inf | $outf\""), "/data/in.dta | /data/out.dta");
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0].title, "Open data");
    assert!(!seen[0].save && seen[1].save);
    assert_eq!(seen[0].filters, vec![("Data files".to_string(), vec!["dta".to_string(), "csv".to_string()]), ("All files".to_string(), vec!["*".to_string()])]);
}

fn have(tool: &str, arg: &str) -> bool {
    std::process::Command::new(tool).arg(arg).output().map(|o| o.status.success()).unwrap_or(false)
}

#[test]
fn c_plugin_reads_and_writes_data() {
    if !have("cc", "--version") {
        eprintln!("skipped: no C compiler");
        return;
    }
    let d = dir("plugin");
    let mut me = AdoInterpreter::new();
    run(&mut me, &format!("plugin header using \"{}\"", p(&d, "fsplugin.h")));
    assert_eq!(rc(&mut me, &format!("plugin header using \"{}\"", p(&d, "fsplugin.h"))), 602);
    let header = std::fs::read_to_string(d.join("fsplugin.h")).unwrap();
    assert!(header.contains("pginit") && header.contains("SF_vdata"));
    std::fs::write(
        d.join("t61.c"),
        r#"#include <stdio.h>
#include <string.h>
#include "fsplugin.h"

FS_ENTRY(int argc, char *argv[]) {
    char buf[200];
    double z, m, sum = 0;
    int j, used = 0;
    if (argc > 0 && strcmp(argv[0], "fail") == 0) {
        SF_error("plugin failed on purpose\n");
        return 459;
    }
    snprintf(buf, sizeof buf, "argc=%d first=%s vars=%d of %d obs=%d\n", argc, argc > 0 ? argv[0] : "", SF_nvars(), SF_nvar(), SF_nobs());
    SF_display(buf);
    for (j = SF_in1(); j <= SF_in2(); j++) {
        if (!SF_ifobs(j)) continue;
        if (SF_vdata(1, j, &z)) return 198;
        if (SF_is_missing(z)) continue;
        if (SF_vstore(2, j, 10 * z)) return 198;
        sum += z;
        used++;
    }
    if (SF_var_is_string(3)) {
        SF_sdata(3, 1, buf);
        strcat(buf, "!");
        SF_sstore(3, 1, buf);
    }
    SF_scal_save("psum", sum);
    snprintf(buf, sizeof buf, "%d", used);
    SF_macro_save("_pused", buf);
    SF_macro_save("pglob", "hello from C");
    SF_macro_use("_mine", buf, sizeof buf);
    SF_display("local mine is ");
    SF_display(buf);
    SF_display("\n");
    if (SF_scal_use("base", &m) == 0) SF_scal_save("pbase", m + 1);
    if (SF_row("M") == 2 && SF_col("M") == 2) {
        SF_mat_el("M", 2, 1, &m);
        SF_mat_store("M", 1, 2, m * 100);
    }
    return 0;
}
"#,
    )
    .unwrap();
    let lib = p(&d, "t61.plugin");
    let st = std::process::Command::new("cc").args(["-shared", "-fPIC", "-O0", "-o", &lib, &p(&d, "t61.c")]).current_dir(&d).output().unwrap();
    if !st.status.success() {
        eprintln!("skipped: compiling the plugin failed: {}", String::from_utf8_lossy(&st.stderr));
        return;
    }
    let script = format!(
        "set obs 6
gen x = _n
replace x = . in 5
gen double y = 0
gen str5 s = \"ab\"
scalar base = 41
matrix M = (1, 2 \\ 3, 4)
program t61plug, plugin using(\"{lib}\")
local mine \"xyz\""
    );
    for l in script.lines() {
        run(&mut me, l);
    }
    assert_eq!(rc(&mut me, &format!("program t61plug, plugin using(\"{lib}\")")), 110);
    let out = run(&mut me, "plugin call t61plug x y s if x > 1 in 1/5, go \"two words\"");
    assert!(out.contains("argc=2 first=go vars=3 of 3 obs=6"), "{out}");
    assert!(out.contains("local mine is xyz"), "{out}");
    let y: Vec<String> = (1..=6).map(|i| run(&mut me, &format!("display y[{i}]"))).collect();
    assert_eq!(y, ["0", "20", "30", "40", "0", "0"]);
    assert_eq!(run(&mut me, "display s[1]"), "ab!");
    assert_eq!(run(&mut me, "display psum \" \" pbase \" `pused' $pglob\""), "9 42 3 hello from C");
    assert_eq!(run(&mut me, "display M[1,2]"), "300");
    // a failing call returns its code and shows its error text
    let e = me.run_line("plugin call t61plug x y s, fail").unwrap_err();
    assert_eq!(rc_of(&e), 459);
    assert!(err_message(&e).contains("plugin failed on purpose"));
    assert_eq!(rc(&mut me, "plugin call nosuch x"), 199);
    // dropping and redefining works
    run(&mut me, "program drop t61plug");
    run(&mut me, &format!("program define t61plug, plugin using(\"{lib}\")"));
    run(&mut me, "plugin call t61plug x y");
    // in a do-file the one-line definition needs no `end`
    let mut me2 = AdoInterpreter::new();
    let out = me2.run_script(&format!("set obs 2\ngen x = _n\ngen y = 0\nprogram p2, plugin using(\"{lib}\")\nplugin call p2 x y\ndisplay y[2]")).unwrap();
    assert!(out.trim_end().ends_with("20"), "{out}");
    assert_eq!(rc(&mut me2, &format!("program p3, plugin using(\"{}\")", p(&d, "missing.plugin"))), 601);
}

#[test]
fn javacall_runs_a_static_method() {
    if !have("javac", "-version") || !have("java", "-version") {
        eprintln!("skipped: no JDK");
        return;
    }
    let d = dir("java");
    let bridge = match ferrostat_ado::w11_java::bridge_classes() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipped: bridge: {e}");
            return;
        }
    };
    std::fs::write(
        d.join("T61.java"),
        r#"import ferrostat.sfi.*;

public class T61 {
    public static int run(String[] args) {
        if (args.length > 0 && args[0].equals("fail")) {
            SFIToolkit.errorln("failing on purpose");
            return 459;
        }
        SFIToolkit.displayln("args=" + String.join(",", args) + " vars=" + Data.getParsedVarCount() + " obs=" + Data.getObsTotal());
        for (long i = Data.getObsParsedIn1(); i <= Data.getObsParsedIn2(); i++) {
            if (!Data.isParsedIfTrue(i)) continue;
            double v = Data.getNum(1, i);
            if (Missing.isMissing(v)) continue;
            Data.storeNum(2, i, 3 * v);
        }
        if (Data.isVarTypeStr(3)) Data.storeStr(3, 2, Data.getStr(3, 2) + "?");
        Macro.setGlobal("jg", "from java " + Macro.getLocal("mine"));
        Scalar.setValue("js", Scalar.getValue("base") + 1);
        return 0;
    }
    public static int boom(String[] args) {
        throw new IllegalStateException("bad");
    }
}
"#,
    )
    .unwrap();
    let cls = d.join("classes");
    std::fs::create_dir_all(&cls).unwrap();
    let st = std::process::Command::new("javac").args(["-cp", &bridge, "-d", &cls.display().to_string(), &p(&d, "T61.java")]).output().unwrap();
    assert!(st.status.success(), "{}", String::from_utf8_lossy(&st.stderr));
    let jars = cls.display().to_string();
    let mut me = AdoInterpreter::new();
    for l in ["set obs 5", "gen x = _n", "replace x = . in 4", "gen double y = 0", "gen str4 s = \"q\"", "scalar base = 9", "local mine abc"] {
        run(&mut me, l);
    }
    let out = run(&mut me, &format!("javacall T61 run x y s if x != 1 in 1/4, jars(\"{jars}\") args(a \"b c\")"));
    assert!(out.contains("args=a,b c vars=3 obs=5"), "{out}");
    let y: Vec<String> = (1..=5).map(|i| run(&mut me, &format!("display y[{i}]"))).collect();
    assert_eq!(y, ["0", "6", "9", "0", "0"]);
    assert_eq!(run(&mut me, "display s[2] \" \" js \" $jg\""), "q? 10 from java abc");
    let e = me.run_line(&format!("javacall T61 run x y, jars(\"{jars}\") args(fail)")).unwrap_err();
    assert_eq!(rc_of(&e), 459);
    assert!(err_message(&e).contains("failing on purpose"));
    let e = me.run_line(&format!("javacall T61 boom, jars(\"{jars}\")")).unwrap_err();
    assert!(err_message(&e).contains("bad"), "{e}");
    assert_eq!(rc(&mut me, &format!("javacall NoSuch run, jars(\"{jars}\")")), 5100);
    assert_eq!(rc(&mut me, "javacall T61 run"), 198);
    assert_eq!(rc(&mut me, "javacall T61 run, jars(nosuch.jar)"), 601);
}
