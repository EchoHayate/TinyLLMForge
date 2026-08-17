# Qwen3.5 Hybrid Prefix Invisible Precommit Design

## Objective

Move every fallible content-interning operation out of visible publication.

`prepare_publication()` currently owns exact clones privately, but
`commit_publication()` still hashes/interns and can fail before visibility.
An all-rank coordinator needs a phase where every rank finishes that work while
the new snapshot remains invisible.

Add:

```text
prepare -> precommit -> finalize
                  \-> rollback
```

## State Machine

```text
empty
  -> prepared
  -> precommitted
  -> finalized

prepared or precommitted
  -> rolled back
```

Only one transaction remains in flight per cache.

## Precommit

`precommit_publication(handle)`:

- validates the exact live handle;
- acquires all canonical intern refs;
- builds the complete immutable snapshot;
- rolls back refs/counters on any failure;
- retains the new snapshot privately;
- does not mutate entries, LRU, visible logical bytes, publishes, replacements,
  or publication commits.

Physical intern storage may increase during precommit. It is private reserved
storage and must be reported separately:

```text
current_precommitted_bytes
current_precommitted_references
publication_precommits
```

Visible `current_bytes` remains the unique physical storage owned by visible
entries. Therefore intern-table physical bytes need separate internal
accounting from visible bytes.

## Finalize

`finalize_publication(handle)` performs no hashing, tensor equality, cloning,
or intern allocation. It installs the already-built snapshot, atomically
replaces an existing entry, updates visible accounting/counters, enforces LRU,
and consumes the transaction.

The existing `commit_publication()` convenience method performs precommit then
finalize.

## Rollback

Rollback from prepared drops clones. Rollback from precommitted releases all
reserved intern refs and restores intern counters to their precommit snapshot.
Visible entries remain unchanged.

## Tests

1. precommit leaves entries and visible bytes unchanged;
2. precommit reserves exact intern refs privately;
3. precommit failure restores refs/counters and stays prepared;
4. rollback from precommitted releases reserved refs;
5. finalize performs zero digest/equality work;
6. finalize publishes exact state and consumes the handle;
7. immediate publish remains compatible.

## Claim Boundary

Passing proves a cache-local non-visible fallible phase and a prepared
non-allocating finalize path. It does not yet prove cross-rank rollback after a
partial finalize or runtime publication.
