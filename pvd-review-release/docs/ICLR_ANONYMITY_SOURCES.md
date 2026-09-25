# ICLR 2027 Source Rules for Anonymous Code Release

This note records the first-party rules used for the review-period release
check. It is a source summary, not legal advice and not a guarantee that a
particular hosting service is anonymous.

## Primary sources

- [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [ICLR Code of Ethics](https://iclr.cc/public/CodeOfEthics)

The pages above were consulted on 2026-09-21. The quotations below are the
relevant rules as published by ICLR; keep the linked pages as the authoritative
versions if they change.

## Rules that apply to this release

### Double-blind anonymity

The Author Guide states:

> “ICLR 2027 is double blind, which means that all submitted papers should be
> anonymous. Any paper where author identity is revealed in either the main text
> or the supplementary material will be desk rejected.”

The repository, archive, README, examples, generated reports, package metadata,
links, and hosting account therefore must not expose author or institution
identity during review. An anonymous code artifact does not make an
identity-revealing URL or account acceptable.

### Anonymous code links

In its “How can we make our code available for reviewing anonymously?” FAQ, the
Author Guide gives three allowed approaches:

1. “Anonymize your code, put it in a .zip file and submit it as supplementary
   materials.”
2. “Make an anonymous repository and put the link in your paper.”
3. After discussion opens, post a link to an anonymous repository in a comment
   directed to the reviewers and area chairs; this keeps the code visible only
   to those reviewers and ACs.

This release follows the same boundary: source history, remotes, archive
metadata, URLs, package metadata, and downloadable assets need an explicit
anonymity review. The FAQ does not certify any particular repository provider.

### Demonstrations and visitor tracking

The Author Guide answers the demonstration-link question as follows:

> “Yes, such a link can be submitted as long as it is completely anonymous.
> Also, make sure that the host website does not track visitors because such
> information can reveal the identity of the reviewer. If your link is found to
> have such issues, it will put your submission at the risk of rejection.”

Any demo, video, hosted notebook, download page, badge, analytics endpoint, or
redirect must therefore be checked for identity leakage and visitor tracking.
This repository does not treat a link as review-safe merely because its visible
page has no author name.

### Reproducibility and code submission

The Author Guide says that source code “can be uploaded as part of the
supplementary material,” that code gives reviewers information “especially for
replicability,” and that ICLR encourages authors to submit code. It also says:

> “It is important that the work published in ICLR is reproducible. Authors are
> strongly encouraged to include a paragraph-long Reproducibility Statement”

and gives an “anonymous downloadable source code” link as an example of what the
statement may reference. This repository therefore states its actual
reproduction scope and missing assets instead of implying that a source-only
artifact reruns paper-scale experiments.

### Accuracy, attribution, licenses, and provenance

The Code of Ethics requires:

> “Findings must be reported accurately and honestly. Researchers must not make
> deliberately false or misleading claims, fabricate or falsify data, or
> misrepresent results. Methods and results should be presented in a way that is
> transparent and reproducible.”

It also says researchers should credit creators and “respect copyrights,
patents, trade secrets, license agreements, and other methods of protecting
authors' works.” For privacy and data handling, it requires that:

> “Data should be used in ways consistent with their licences.”

The same section calls for precautions against re-identification, unauthorized
collection or disclosure, and requires “understanding the provenance of the
data.” The release must consequently retain third-party notices, record asset
source/revision/digest and license status, avoid publishing private checkpoints
or simulator data, and label historical or partial reproduction evidence
honestly.

## Release-side interpretation

These rules establish review constraints, not an ICLR approval or a claim that
the release account is anonymous. Before sharing a review URL or archive, run the
repository anonymity audit and perform a manual review of account metadata,
commit/archive metadata, binary metadata, links, network requests, analytics,
third-party licenses, asset provenance, and any external download gate.
