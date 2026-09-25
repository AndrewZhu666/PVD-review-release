# Anonymous Release Checklist

ICLR double-blind review requires submitted paper material to avoid revealing
author identity. The official guidelines also require external review links to
be completely anonymous and warn against hosting that tracks visitors. This
checklist translates that boundary into repository hygiene; it does not replace
the current conference rules or advice from the program chairs.

Official references:

- <https://iclr.cc/Conferences/2027/AuthorGuidelines>
- <https://iclr.cc/public/CodeOfEthics>

## Automated check

From the repository root:

```bash
python scripts/audit_anonymity.py .
```

The scanner checks file and directory names, UTF-8 text, symlink targets, and,
when the root is a Git repository, history and remotes. It skips local cache
directories for speed but performs a second pass over any tracked files inside
those directories. It rejects common:

- per-user and machine-mounted absolute paths;
- cluster host and internal experiment identifiers;
- experiment-tracking metadata;
- non-attribution email addresses and identity metadata;
- private-network addresses, URL credentials, and private key material;
- non-anonymous Git author/committer identities;
- absolute symlink targets and oversized files that were not inspected.

JSON output is available for CI:

```bash
python scripts/audit_anonymity.py . --format json
```

Findings redact the matched value. The process exits `0` only when no finding is
present, `1` for findings, and `2` for scanner errors.

Upstream email addresses are allowed only in license-like files below
`third_party/`, where removing attribution could violate a license. Other rules
still apply to those files.

## Manual content review

- Confirm README, docs, comments, examples, configs, and expected output contain
  no author names, affiliations, personal usernames, lab names, personal URLs,
  acknowledgements, or self-identifying prose.
- Inspect notebooks, PDFs, images, videos, archives, checkpoints, and generated
  reports for embedded author, application, filesystem, camera, and EXIF data.
- Inspect package metadata, container labels, model cards, dataset cards,
  citations, security contacts, ownership files, and issue templates.
- Search for private bucket names, registry namespaces, internal domains,
  ticket identifiers, job IDs, process logs, and credential variable names.
- Confirm example failures and tracebacks do not reveal a local checkout path.
- Confirm no secret, token, cookie, credential, or private key is present even
  if it would not identify an author.

## Git and archive review

- Create the review repository from the audited file tree with a new anonymous
  history; do not publish the development repository or a fork relationship.
- Use an anonymous author and committer identity with a non-routable placeholder
  email domain.
- Remove original remotes, signed commits, tags, branches, pull requests, issue
  links, submodule endpoints, and large-file storage endpoints that disclose
  provenance.
- Inspect commit messages and timestamps for identity or private infrastructure.
- Build the final ZIP from the audited file allowlist, list every member, and inspect archive
  comments, owner fields, symlinks, hidden files, and nested archives.
- Normalize archive ownership before upload so local account names cannot leak
  through tar headers (for example, use
  `tar --owner=0 --group=0 --numeric-owner` when creating a tar archive);
  verify the resulting listing contains no workstation-specific user or group
  names.
- Run the scanner on the exact unpacked ZIP, not only on a working tree.

## Hosting and reviewer privacy

- The repository account, organization, URL, avatar, and profile must not be
  linkable to an author or institution.
- Do not use author-controlled analytics, beacons, short-link statistics,
  download forms, access-request workflows, or callbacks.
- Do not include remote badges or images that create a request when README is
  rendered.
- Required assets must be downloadable without contacting an author or exposing
  reviewer identity. Third-party gated access must be described as a third-party
  process, never as an author approval flow.
- Review the hosting platform's visitor analytics and access logs against the
  current conference rule; platform defaults are not automatically compliant.

## Third-party identity and licenses

Do not delete upstream names, citations, copyrights, license texts, or notices
to make the repository look anonymous. Third-party attribution is not an author
declaration. Keep it in a clearly separated `third_party/` area and follow
[THIRD_PARTY.md](THIRD_PARTY.md).

The project license is a separate rights decision. Do not invent an anonymous
copyright holder or remove an upstream notice. Obtain institutional/legal
approval for the review-period and post-review license wording.

## Final gate

The automated scan is necessary but insufficient. Release only after a person
who did not prepare the files has reviewed the exact repository URL and exact
archive from the perspective of a reviewer, including network requests made by
rendered pages and downloads.
