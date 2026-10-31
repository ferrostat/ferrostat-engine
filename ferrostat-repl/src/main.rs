use std::env;
use std::io::{self, BufRead, IsTerminal, Write};

use ferrostat_ado::interpreter::{err_message, rc_of, AdoInterpreter};

fn print_result(interp: &mut AdoInterpreter, res: Result<String, String>) -> bool {
    if res.is_err() {
        let partial = std::mem::take(&mut interp.partial_output);
        if !partial.trim().is_empty() {
            println!("{}", partial.trim_end());
        }
    }
    match res {
        Ok(out) => {
            let exit = out.ends_with("__EXIT__");
            let mut text = out.trim_end_matches("__EXIT__").trim_end().to_string();
            // `cls` clears a terminal; redirected output keeps everything
            let cls = ferrostat_ado::interpreter::CLS;
            if let Some(i) = text.rfind(cls) {
                text = if io::stdout().is_terminal() {
                    print!("\x1b[2J\x1b[H");
                    text[i + cls.len()..].trim_start_matches('\n').to_string()
                } else {
                    text.lines().filter(|l| !l.contains(cls)).collect::<Vec<_>>().join("\n")
                };
            }
            if !text.is_empty() {
                println!("{text}");
            }
            exit
        }
        Err(e) => {
            let msg = err_message(&e);
            if !msg.trim().is_empty() {
                eprintln!("{}", msg.trim_end());
            }
            eprintln!("r({});", rc_of(&e));
            false
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    // every word accepted as a command (with abbreviations), one per line
    if args.iter().any(|a| a == "--list-commands") {
        println!("{}", ferrostat_ado::session::Session::new().command_words().join("\n"));
        return;
    }
    // Static check: ferrostat --check file.do [more files] [--strict]
    if let Some(i) = args.iter().position(|a| a == "--check") {
        let strict = args.iter().any(|a| a == "--strict");
        let files: Vec<String> = args[i + 1..].iter().filter(|a| !a.starts_with("--")).cloned().collect();
        if files.is_empty() {
            eprintln!("usage: ferrostat --check file.do [more files] [--strict]");
            std::process::exit(198);
        }
        let interp = AdoInterpreter::new();
        let mut words = interp.engine().command_words();
        words.extend(ferrostat_ado::progcmd::INTERP_COMMANDS.iter().map(|s| s.to_string()));
        let (report, (errors, warnings, _)) = ferrostat_ado::lint::check_files(&files, words, &[]);
        println!("{report}");
        std::process::exit(if errors > 0 || strict && warnings > 0 { 1 } else { 0 });
    }
    let mut interp = AdoInterpreter::new();

    // Batch mode: ferrostat --do file.do [args]  (also: ferrostat -b do file.do)
    let batch = args.iter().position(|a| a == "--do" || a == "do");
    if let Some(i) = batch {
        let Some(path) = args.get(i + 1) else {
            eprintln!("usage: ferrostat --do file.do [args]");
            std::process::exit(198);
        };
        let rest = args[i + 2..].join(" ");
        interp.set_echo(true);
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("file {path} not found\nr(601);");
                std::process::exit(601);
            }
        };
        let _ = rest;
        let res = interp.run_script(&text);
        let failed = res.is_err();
        let code = res.as_ref().err().map(|e| rc_of(e)).unwrap_or(0);
        print_result(&mut interp, res);
        if failed {
            std::process::exit(code.max(1));
        }
        return;
    }

    println!("Ferrostat {} — statistics engine", env!("CARGO_PKG_VERSION"));
    println!("Type a command, or 'exit' to quit.");
    ferrostat_ado::w11_ui::terminal_attach(&mut interp);
    let stdin = io::stdin();
    loop {
        print!("{}", if interp.is_pending() { "> " } else { ". " });
        io::stdout().flush().unwrap();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let r = interp.run_line(line.trim_end_matches(['\n', '\r']));
        let done = print_result(&mut interp, r);
        // editor / web browser requests (doedit, view browse)
        ferrostat_ado::w11_ui::terminal_service(&mut interp).iter().filter(|m| !m.is_empty()).for_each(|m| println!("{m}"));
        if done {
            break;
        }
    }
}
