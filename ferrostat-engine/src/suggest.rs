//! "Did you mean …?" hints for misspelt commands, variables and options.

/// Optimal-string-alignment distance (Levenshtein plus adjacent swaps),
/// case-insensitive.
pub fn distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.to_lowercase().chars().collect();
    let b: Vec<char> = b.to_lowercase().chars().collect();
    let (n, m) = (a.len(), b.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=m {
        d[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[n][m]
}

/// The candidate closest to `word`, if it is close enough to be a likely
/// typo: within a third of the word's length (at least 1), or a candidate
/// that is a prefix of the word (`regression` for `regress`). Ties go to the
/// first candidate given.
pub fn closest<'a, I>(word: &str, candidates: I) -> Option<String>
where
    I: IntoIterator<Item = &'a str>,
{
    let w = word.to_lowercase();
    if w.is_empty() {
        return None;
    }
    let limit = (w.chars().count() / 3).max(1);
    let mut best: Option<(usize, &str)> = None;
    for c in candidates {
        let cl = c.to_lowercase();
        if cl == w {
            return None;
        }
        // a longer word starting with a whole command name (`regression`)
        let d = if cl.len() >= 3 && w.starts_with(&cl) { 1 } else { distance(&w, &cl) };
        if d <= limit && best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, c));
        }
    }
    best.map(|(_, c)| c.to_string())
}

/// The error text with a hint line added before the `r(#);` line.
pub fn with_hint(err: &str, hint: &str) -> String {
    match err.rfind("\nr(") {
        Some(i) => format!("{}\n{hint}{}", &err[..i], &err[i..]),
        None => format!("{err}\n{hint}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typos_and_prefixes() {
        let cmds = ["regress", "rename", "replace", "summarize", "tabulate", "generate"];
        assert_eq!(closest("regression", cmds).as_deref(), Some("regress"));
        assert_eq!(closest("sumarize", cmds).as_deref(), Some("summarize"));
        assert_eq!(closest("tabluate", cmds).as_deref(), Some("tabulate"));
        assert_eq!(closest("xyzzy", cmds), None);
        assert_eq!(closest("regress", cmds), None);
    }

    #[test]
    fn hint_goes_before_return_code() {
        assert_eq!(with_hint("bad\nr(199);", "(did you mean x?)"), "bad\n(did you mean x?)\nr(199);");
    }
}
