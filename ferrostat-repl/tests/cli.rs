//! The command-line binary in batch mode.

use std::process::Command;

fn run_do(name: &str, script: &str) -> std::process::Output {
    let dir = std::env::temp_dir().join(format!("ferrostat-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{name}.do"));
    std::fs::write(&path, script).unwrap();
    Command::new(env!("CARGO_BIN_EXE_ferrostat")).arg("--do").arg(&path).output().unwrap()
}

#[test]
fn batch_do_file_runs() {
    let out = run_do("ok", "set obs 3\ngenerate x = _n\nsummarize x\ndisplay \"done\"\n");
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "{text}");
    assert!(text.contains("done") && text.contains(". summarize x"), "{text}");
}

#[test]
fn batch_errors_set_exit_code() {
    let out = run_do("err", "summarize nosuchvar\n");
    assert_eq!(out.status.code(), Some(111));
    assert!(String::from_utf8_lossy(&out.stderr).contains("r(111);"));
}
