//! Do-file linter (`ferrostat --check`, `lint using`) and the hover side
//! channel of on-screen graphs.

use ferrostat_ado::lint::{self, Commands, Diag, Severity};
use ferrostat_tests::*;

fn check(text: &str) -> Vec<Diag> {
    let words = ferrostat_ado::session::Session::new().command_words();
    lint::check_text("inline.do", text, &Commands::new(words))
}

fn has(d: &[Diag], sev: Severity, line: usize, needle: &str) -> bool {
    d.iter().any(|x| x.severity == sev && x.line == line && x.message.contains(needle))
}

fn errors(d: &[Diag]) -> Vec<&Diag> {
    d.iter().filter(|x| x.severity == Severity::Error).collect()
}

#[test]
fn unbalanced_braces_and_blocks() {
    let d = check("foreach v in a b {\n  di \"`v'\"\n");
    assert!(has(&d, Severity::Error, 1, "unbalanced \"{\""), "{d:?}");
    assert_eq!(d.iter().find(|x| x.severity == Severity::Error).unwrap().col, 18);
    let d = check("di 1\n}\n");
    assert!(has(&d, Severity::Error, 2, "unexpected \"}\""), "{d:?}");
    let d = check("if 1 {\n  di 1\n} else {\n  di 2\n}\n");
    assert!(errors(&d).is_empty(), "{d:?}");
    let d = check("program define f\n  if 1 {\n    di 1\nend\n");
    assert!(has(&d, Severity::Error, 2, "the program ends at line 4"), "{d:?}");
    let d = check("program define f\n  di 1\n");
    assert!(has(&d, Severity::Error, 1, "program f is not closed"), "{d:?}");
    let d = check("di 1\nend\n");
    assert!(has(&d, Severity::Error, 2, "\"end\" without program"), "{d:?}");
    let d = check("mata:\n x = 1\n");
    assert!(has(&d, Severity::Error, 1, "mata block"), "{d:?}");
    // braces and strings inside mata and python blocks are not do-file code
    let d = check("mata:\nfunction f() {\n return(\"{\")\n}\nend\npython:\nprint(\"\"\"x\"\"\")\nend\n");
    assert!(errors(&d).is_empty(), "{d:?}");
}

#[test]
fn strings_comments_and_continuations() {
    let d = check("di \"abc\n");
    assert!(has(&d, Severity::Error, 1, "unterminated string"), "{d:?}");
    let d = check("di `\"a \"quoted\" b\"'\ndi `\"open\n");
    assert!(!has(&d, Severity::Error, 1, "unterminated"), "{d:?}");
    assert!(has(&d, Severity::Error, 2, "unterminated compound quote"), "{d:?}");
    let d = check("di 1\n/* never\nclosed\n");
    assert!(has(&d, Severity::Warning, 2, "unterminated \"/*\" comment"), "{d:?}");
    // comments may hide braces; continuations join lines
    let d = check("/* { */\ndi 1 // {\n* {\nforeach v in a ///\n   b {\n  di \"`v'\"\n}\n");
    assert!(errors(&d).is_empty(), "{d:?}");
    let d = check("#delimit ;\nforeach v in a\n b {;\n di \"`v'\";\n};\n#delimit cr\ndi 1\n");
    assert!(errors(&d).is_empty(), "{d:?}");
    // a string spanning a `//` stays a string
    let d = check("di \"http://example.org\"\n");
    assert!(d.is_empty(), "{d:?}");
}

#[test]
fn unknown_commands_with_suggestions() {
    let d = check("sysuse auto\ntabluate price mpg\n");
    let e = errors(&d);
    assert_eq!(e.len(), 1, "{d:?}");
    assert_eq!((e[0].line, e[0].col), (2, 1));
    assert!(e[0].message.contains("unknown command \"tabluate\"") && e[0].message.contains("did you mean"), "{}", e[0].message);
    // under capture only a warning; prefixes are looked through
    let d = check("capture nosuchcmd x\nquietly bysort g: nosuchcmd2\nby g: summarize x\n");
    assert!(has(&d, Severity::Warning, 1, "nosuchcmd"), "{d:?}");
    assert!(has(&d, Severity::Error, 2, "nosuchcmd2"), "{d:?}");
    assert!(!d.iter().any(|x| x.line == 3), "{d:?}");
    // programs defined in the file, abbreviations and one-line if
    let d = check("program define myprog\n  di \"hi\"\nend\nmyprog\nqui su x\nif 1 == 1 di \"yes\"\nif 1 == 1 nosuch\n");
    assert!(errors(&d).iter().all(|x| x.line == 7), "{d:?}");
    assert!(has(&d, Severity::Error, 7, "nosuch"), "{d:?}");
    // capture without checking _rc is fine; else without if is not
    let d = check("capture drop x\nelse {\n di 1\n}\n");
    assert!(has(&d, Severity::Error, 2, "\"else\" without \"if\""), "{d:?}");
    assert_eq!(errors(&d).len(), 1, "{d:?}");
    // a top-level exit ends the file
    let d = check("di 1\nexit\nthis is just text {\n");
    assert!(d.is_empty(), "{d:?}");
}

#[test]
fn local_macros_defined_and_used() {
    let text = "\
local a 1
local unused 2
di `a' `b'
foreach v of varlist x y {
    di \"`v'\"
}
forvalues i = 1/3 {
    di `i'
}
levelsof x, local(levels)
di \"`levels' `c(pwd)' `r(N)' `1'\"
tempvar t
gen `t' = 1
local `a'_x 5
di `1_x'
program define p
    syntax varlist [if] [, Detail Level(real 95) noCONStant *]
    args first
    di \"`varlist' `if' `detail' `level' `constant' `options' `first' `0'\"
    di \"`a'\"
    marksample touse
    gettoken w rest : varlist
    di \"`touse' `w' `rest'\"
end
";
    let d = check(text);
    assert!(errors(&d).is_empty(), "{d:?}");
    let warns: Vec<&Diag> = d.iter().filter(|x| x.severity == Severity::Warning).collect();
    // `b' is undefined; `a' is not visible inside the program
    assert_eq!(warns.len(), 2, "{d:?}");
    assert!(has(&d, Severity::Warning, 3, "`b'"), "{d:?}");
    assert_eq!(d.iter().find(|x| x.message.contains("`b'")).unwrap().col, 8);
    assert!(has(&d, Severity::Warning, 20, "`a' is used but never defined in this program"), "{d:?}");
    assert!(has(&d, Severity::Note, 2, "`unused' is defined but never used"), "{d:?}");
    assert_eq!(d.iter().filter(|x| x.severity == Severity::Note).count(), 1, "{d:?}");
}

#[test]
fn lint_command_and_report_format() {
    let dir = temp_dir("lint");
    let path = dir.join("bad.do");
    std::fs::write(&path, "foreach v in a {\n  di `v'\ntabluate y x\n").unwrap();
    let p = path.display().to_string();
    let mut s = Session::new();
    let out = s.run(&format!("lint using \"{p}\""));
    assert!(out.contains(&format!("{p}:1:16: error: unbalanced \"{{\" opened here")), "{out}");
    assert!(out.contains(&format!("{p}:3:1: error: unknown command \"tabluate\"")), "{out}");
    assert!(out.trim_end().ends_with("1 file checked: 2 errors, 0 warnings, 0 notes"), "{out}");
    assert_eq!(s.scalar("r(errors)"), 2.0);
    assert_eq!(s.scalar("r(warnings)"), 0.0);
}

#[test]
fn hover_side_channel_for_scatter() {
    let mut s = ferrostat_ado::session::Session::new();
    s.run_script("clear\ninput x y str3 name\n1 2 a\n2 4 b\n3 1 c\n. 5 d\nend\nscatter y x, mlabel(name)");
    let (svg, json) = s.graph_svg_hover(400, 300).expect("graph");
    // the drawing itself carries no metadata or scripts
    assert_eq!(svg, s.graph_svg(400, 300).unwrap());
    assert!(!svg.contains("<script") && !svg.contains("data-"), "svg has metadata");
    assert!(json.starts_with("{\"w\":400,\"h\":300,\"panels\":[["), "{json}");
    // points: [px, py, x, y, obs, series, label]; the missing x is not drawn
    assert_eq!(json.matches(",0,\"").count(), 3, "{json}");
    assert!(json.contains(",1,2,1,0,\"a\"]") && json.contains(",3,1,3,0,\"c\"]"), "{json}");
    assert!(json.contains("\"series\":[{\"name\":\"y\",\"x\":\"x\",\"panel\":0,\"line\":[]}]"), "{json}");
    // hit-testing in Rust (desktop) gives the same readout
    let g = s.graph().unwrap().clone();
    let (_, map) = ferrostat_cmd::hover::render_svg_hover(&g, 400, 300).unwrap();
    let p = &map.points[2];
    let text = map.hit(p.px + 2, p.py - 1).unwrap();
    assert_eq!(text, "y = 1, x = 3 (obs 3) [c]");
    assert!(map.hit(0, 0).is_none());
    // exported files stay clean
    let dir = temp_dir("hover_export");
    let out = dir.join("g.svg").display().to_string();
    s.submit(&format!("graph export \"{out}\", replace"));
    let exported = std::fs::read_to_string(&out).unwrap();
    assert!(!exported.contains("<script") && !exported.contains("data-"));
}

#[test]
fn hover_lines_and_bars() {
    let mut s = ferrostat_ado::session::Session::new();
    s.run_script("clear\nset obs 5\ngen t = 6 - _n\ngen v = t * t\nline v t");
    let g = s.graph().unwrap().clone();
    let (_, map) = ferrostat_cmd::hover::render_svg_hover(&g, 400, 300).unwrap();
    let line = &map.series[0].line;
    assert_eq!(line.len(), 5);
    // sorted by x, observation numbers follow their points
    assert_eq!((line[0].x, line[0].obs), (1.0, 5));
    // anywhere in the plot shows the nearest vertex in x
    let q = &line[1];
    let text = map.hit(q.px + 3, map.panels[0][1] + 5).unwrap();
    assert!(text.starts_with("v = 4, t = 2 (obs 4)"), "{text}");
    s.run_script("clear\ninput g w\n1 3\n2 7\nend\ngraph bar (mean) w, over(g)");
    let g = s.graph().unwrap().clone();
    let (_, map) = ferrostat_cmd::hover::render_svg_hover(&g, 400, 300).unwrap();
    assert_eq!(map.bars.len(), 2, "{:?}", map.bars);
    let b = &map.bars[1];
    let text = map.hit((b.rect[0] + b.rect[2]) / 2, (b.rect[1] + b.rect[3]) / 2).unwrap();
    assert!(text.starts_with("2: ") && text.ends_with("= 7"), "{text}");
}
