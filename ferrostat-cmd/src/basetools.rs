//! Registration of the base post-estimation tools and small statistics
//! commands (lrtest, nlcom, pwcompare, estat classification, brier, ...).

use crate::commands::CommandHandler;

pub fn handlers() -> Vec<(&'static str, Box<dyn CommandHandler>)> {
    let mut v = crate::postest2::handlers();
    v.extend(crate::binpost::handlers());
    v.extend(crate::pwcompare::handlers());
    v.extend(crate::nlest::handlers());
    v.extend(crate::smallstat::handlers());
    v.extend(crate::smallstat2::handlers());
    v.extend(crate::pk::handlers());
    v.extend(crate::pss::handlers());
    v
}
