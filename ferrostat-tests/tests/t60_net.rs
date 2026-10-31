//! Network commands (`net`, `ssc`, `ado`, `adoupdate`, `help` for
//! installed help files, `copy`/`use`/`import` from http:// addresses) and
//! autoloading of installed ado-files. No internet access is needed: the
//! package site is a local directory, and the HTTP path is served by a tiny
//! server thread inside the test.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};

use ferrostat_ado::w10_net::toc_file_name;
use ferrostat_tests::*;

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ferrostat_t60_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write(dir: &Path, name: &str, text: &str) {
    let p = dir.join(name);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, text).unwrap();
}

/// A package site with a table of contents, two packages, a subdirectory
/// and a package named like a natively provided one.
fn site(root: &Path) -> PathBuf {
    let s = root.join("site");
    write(&s, &toc_file_name(), "v 3\nd Test site for the package commands\nt sub More packages\np hello Says hello\np egdemo An egen function\np fre A community frequency table\n");
    write(&s, "hello.pkg", "v 3\nd 'HELLO': module to say hello\nd Says hello.\nd Distribution-Date: 20250101\nd Author: A. Tester, support at test@@example.org\nf hello.ado\nf hello.sthlp\nf hello_data.csv\n");
    write(
        &s,
        "hello.ado",
        "*! version 1.0.0  01jan2025\nprogram define hello\n    version 9\n    syntax [anything] [, Loud]\n    if \"`loud'\" != \"\" {\n        display \"HELLO `anything'\"\n    }\n    else display \"hello `anything'\"\n    hello_sub\nend\nprogram define hello_sub\n    display \"(sub)\"\nend\n",
    );
    write(
        &s,
        "hello.sthlp",
        "{smcl}\n{* *! version 1.0.0}{...}\n{title:Title}\n\n{phang}\n{bf:hello} {hline 2} Say hello{p_end}\n\n{title:Syntax}\n\n{p 8 15 2}\n{cmd:hello} [{it:text}] [{cmd:,} {opt l:oud}]\n\n{synoptset 12}{...}\n{synopthdr}\n{synoptline}\n{synopt:{opt l:oud}}shout the greeting{p_end}\n{synoptline}\n\n{title:Author}\n\n{pstd}A. Tester, see {browse \"http://example.org\":the site}.{p_end}\n",
    );
    write(&s, "hello_data.csv", "a,b\n1,2\n3,4\n");
    write(&s, "egdemo.pkg", "v 3\nd 'EGDEMO': egen function doubling an expression\nf _gtwice.ado\n");
    write(&s, "_gtwice.ado", "program define _gtwice\n    syntax newvarname =/exp [if] [in]\n    quietly generate `typlist' `varlist' = 2*(`exp') `if' `in'\nend\n");
    write(&s, "fre.pkg", "v 3\nd 'FRE': community frequency table (test copy)\nf fre.ado\n");
    write(&s, "fre.ado", "program define fre\n    display \"community fre for `0'\"\nend\n");
    write(&s, &format!("sub/{}", toc_file_name()), "v 3\nd Subdirectory\np deep A package in a subdirectory\n");
    write(&s, "sub/deep.pkg", "v 3\nd 'DEEP': deep package\nf deep.ado\n");
    write(&s, "sub/deep.ado", "program deep\n    display \"deep down\"\nend\n");
    s
}

fn session(root: &Path) -> Session {
    let mut s = Session::new();
    s.run(&format!("sysdir set PLUS \"{}\"", root.join("plus").display()));
    s.run(&format!("cd \"{}\"", root.display()));
    s
}

#[test]
fn net_from_describe_install_uninstall() {
    let root = scratch("local");
    let site = site(&root);
    let mut s = session(&root);
    let out = s.run(&format!("net from \"{}\"", site.display()));
    assert!(out.contains("Test site") && out.contains("hello") && out.contains("DIRECTORIES") && out.contains("PACKAGES"), "{out}");
    let out = s.run("net describe hello");
    assert!(out.contains("'HELLO': module to say hello") && out.contains("INSTALLATION FILES") && out.contains("hello.ado") && out.contains("ANCILLARY FILES") && out.contains("hello_data.csv"), "{out}");
    assert!(out.contains("test@example.org"), "{out}");
    s.fails("hello");
    let out = s.run("net install hello");
    assert!(out.contains("installation complete"), "{out}");
    let plus = root.join("plus");
    assert!(plus.join("h/hello.ado").exists() && plus.join("h/hello.sthlp").exists());
    assert!(!root.join("hello_data.csv").exists());
    assert!(plus.join("ferrostat.trk").exists());
    // the installed command autoloads, with its subprogram
    let out = s.run("hello world");
    assert!(out.contains("hello world") && out.contains("(sub)"), "{out}");
    assert!(s.run("hello there, loud").contains("HELLO there"));
    assert!(s.run("which hello").contains("hello.ado"));
    // installing twice: identical files are fine, changed ones need replace
    assert!(s.run("net install hello").contains("identical"));
    std::fs::write(plus.join("h/hello.ado"), std::fs::read_to_string(site.join("hello.ado")).unwrap() + "\n* edited\n").unwrap();
    s.fails("net install hello");
    assert!(s.run("net install hello, replace").contains("installation complete"));
    // ancillary files
    let out = s.run("net get hello");
    assert!(out.contains("hello_data.csv") && root.join("hello_data.csv").exists(), "{out}");
    // help renders the installed SMCL help file
    let out = s.run("help hello");
    assert!(out.contains("hello -- Say hello") && out.contains("shout the greeting") && out.contains("the site"), "{out}");
    assert!(!out.contains("{"), "{out}");
    // ado dir / describe / uninstall
    let out = s.run("ado dir");
    assert!(out.contains("package hello from") && out.contains("'HELLO'"), "{out}");
    let out = s.run("ado describe hello");
    assert!(out.contains("h/hello.ado") && out.contains("INSTALLED ON"), "{out}");
    let out = s.run("ado uninstall hello");
    assert!(out.contains("uninstalled"), "{out}");
    assert!(!plus.join("h/hello.ado").exists());
    s.fails("hello");
    assert!(s.run("ado dir").contains("no packages"));
    s.fails("ado uninstall hello");
    // net cd, packages in subdirectories, from()
    let out = s.run("net cd sub");
    assert!(out.contains("deep"), "{out}");
    s.run("net install deep");
    assert!(s.run("deep").contains("deep down"));
    assert!(plus.join("d/deep.ado").exists());
    let out = s.run("net query");
    assert!(out.contains("sub"), "{out}");
    s.run(&format!("net install hello, from(\"{}\") all", site.display()));
    assert!(s.run("ado dir").contains("hello"));
    // egen functions from ado-files
    s.run("net cd ..");
    s.run("net install egdemo");
    s.run("clear");
    s.run("set obs 3");
    s.run("gen x = _n");
    s.run("egen y = twice(x)");
    assert_eq!(s.nums("y"), vec![2.0, 4.0, 6.0]);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn native_packages_kept_unless_nonative() {
    let root = scratch("native");
    let site = site(&root);
    let mut s = session(&root);
    s.run(&format!("net from \"{}\"", site.display()));
    let out = s.run("net install fre");
    assert!(out.contains("natively"), "{out}");
    assert!(!root.join("plus/f/fre.ado").exists());
    s.run("clear");
    s.run("set obs 4");
    s.run("gen x = mod(_n, 2)");
    assert!(!s.run("fre x").contains("community fre"));
    let out = s.run("net install fre, nonative");
    assert!(out.contains("installation complete"), "{out}");
    assert!(s.run("fre x").contains("community fre for x"));
    s.run("ado uninstall fre");
    assert!(!s.run("fre x").contains("community fre"));
    // set ado_native off: every installed ado-file wins
    s.run("set ado_native off");
    s.run("net install fre");
    assert!(s.run("fre x").contains("community fre"));
    s.run("set ado_native on");
    s.run("ado uninstall fre");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn adoupdate_detects_new_distribution() {
    let root = scratch("update");
    let site = site(&root);
    let mut s = session(&root);
    s.run(&format!("net install hello, from(\"{}\")", site.display()));
    let out = s.run("adoupdate");
    assert!(out.contains("up to date") && out.contains("no packages require updating"), "{out}");
    let pkg = std::fs::read_to_string(site.join("hello.pkg")).unwrap().replace("20250101", "20260101");
    std::fs::write(site.join("hello.pkg"), pkg).unwrap();
    std::fs::write(site.join("hello.ado"), "program define hello\n    display \"hello v2\"\nend\n").unwrap();
    let out = s.run("adoupdate");
    assert!(out.contains("not up to date") && out.contains("adoupdate, update"), "{out}");
    let out = s.run("adoupdate hello, update");
    assert!(out.contains("updated"), "{out}");
    assert!(s.run("hello").contains("hello v2"));
    assert!(s.run("ado update").contains("no packages require updating"));
    let _ = std::fs::remove_dir_all(&root);
}

/// Serve `dir` over HTTP/1.0 on localhost; returns the base URL.
fn serve(dir: PathBuf) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut first = String::new();
            let _ = reader.read_line(&mut first);
            loop {
                let mut h = String::new();
                if reader.read_line(&mut h).unwrap_or(0) == 0 || h.trim().is_empty() {
                    break;
                }
            }
            let path = first.split_whitespace().nth(1).unwrap_or("/").trim_start_matches('/').to_string();
            match std::fs::read(dir.join(&path)) {
                Ok(body) => {
                    let _ = write!(stream, "HTTP/1.0 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                    let _ = stream.write_all(&body);
                }
                Err(_) => {
                    let _ = write!(stream, "HTTP/1.0 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                }
            }
        }
    });
    format!("http://127.0.0.1:{port}")
}

#[test]
fn http_copy_use_import_and_install() {
    let root = scratch("http");
    let site = site(&root);
    let mut s = session(&root);
    // a dataset to serve
    s.run("clear");
    s.run("set obs 3");
    s.run("gen v = _n * 10");
    s.run(&format!("save \"{}\"", site.join("tiny.dta").display()));
    let base = serve(site.clone());
    s.run(&format!("copy {base}/hello_data.csv got.csv"));
    assert_eq!(std::fs::read_to_string(root.join("got.csv")).unwrap(), "a,b\n1,2\n3,4\n");
    s.fails(&format!("copy {base}/hello_data.csv got.csv"));
    s.run(&format!("copy {base}/hello_data.csv got.csv, replace"));
    s.fails(&format!("copy {base}/no_such_file.csv other.csv"));
    s.run(&format!("use {base}/tiny.dta, clear"));
    assert_eq!(s.nums("v"), vec![10.0, 20.0, 30.0]);
    s.run(&format!("import delimited using {base}/hello_data.csv, clear"));
    assert_eq!(s.nums("b"), vec![2.0, 4.0]);
    s.run(&format!("insheet using {base}/hello_data.csv, clear"));
    assert_eq!(s.nums("a"), vec![1.0, 3.0]);
    let out = s.run(&format!("net from {base}"));
    assert!(out.contains("Test site"), "{out}");
    let out = s.run("net install hello");
    assert!(out.contains("installation complete"), "{out}");
    assert!(s.run("hello web").contains("hello web"));
    let out = s.run("ado dir");
    assert!(out.contains(&format!("from {base}")), "{out}");
    s.fails("net describe nosuchpkg");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn netio_settings() {
    let mut s = Session::new();
    s.run("set httpproxyhost proxy.example.org");
    s.run("set httpproxyport 3128");
    s.run("set httpproxy on");
    let cfg = ferrostat_core::net::config();
    assert_eq!(ferrostat_core::net::proxy_url(&cfg).as_deref(), Some("http://proxy.example.org:3128"));
    s.run("set timeout1 5");
    assert_eq!(ferrostat_core::net::config().timeout1, 5);
    s.fails("set httpproxy maybe");
    s.fails("set httpproxyport 0");
    s.run("set httpproxy off");
    s.run("set timeout1 30");
    assert_eq!(ferrostat_core::net::proxy_url(&ferrostat_core::net::config()), None);
    assert!(s.run("display c(httpproxy)").contains("off"));
    s.run("set fredkey abc123");
    // no location: describe needs net from or from()
    s.fails("net describe something");
}
