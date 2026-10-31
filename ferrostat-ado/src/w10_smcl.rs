//! Help files (`.sthlp`, `.hlp`) as plain text for `help`/`whelp` and
//! `ssc type`: SMCL goes through the shared renderer
//! (`ferrostat_cmd::w11::smcl`); old-style help files (plain text with
//! `^bold^` and `@link@` markers) are cleaned up here.

use ferrostat_cmd::w11::smcl;

/// Render a help file as plain text.
pub fn render_help(src: &str) -> String {
    if src.trim_start().starts_with("{smcl}") || src.contains("{p_end}") || src.contains("{title:") {
        return smcl::to_text(src);
    }
    src.lines()
        .map(|l| if l.trim() == ".-" { "-".repeat(smcl::WIDTH) } else { l.replace(['^', '@'], "").trim_end().to_string() })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smcl_and_old_help() {
        let t = render_help("{smcl}\n{title:Title}\n\n{pstd}{cmd:demo} {hline 2} a demo{p_end}\n");
        assert!(t.contains("Title") && t.contains("demo") && !t.contains('{'), "{t}");
        assert_eq!(render_help("^demo^ -- see @help@"), "demo -- see help");
    }
}
