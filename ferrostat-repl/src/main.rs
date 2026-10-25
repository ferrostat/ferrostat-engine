use std::env;
use std::io::{self, BufRead, Write};

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
            let text = out.trim_end_matches("__EXIT__").trim_end();
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
        if print_result(&mut interp, r) {
            break;
        }
    }
}
