# Working in this repository

The highest value is code a human can read. Everything below serves that.

## Readability

Concise and clear, and where they pull against each other, say which one you traded away.

Names carry the meaning. Variables and axis labels should say what a thing *is* in the language of
the problem — `onlyRemaining`, `wantTheSame`, `Candidate` — not what it is mechanically. A good name
removes the need for a comment.

Order tells the story. Within an expression, put the terms in the order the reader thinks them:
`minimum(aMin + bNext, bMin + aNext)` — first a keeps its node and b takes its next, then the other
way round. Sorting the same terms into `aMin + bNext, aNext + bMin` makes a tidier-looking symmetry
in a and b, but that symmetry is not what the code is about, and the reader has to undo it.

## Prompt baggage

A comment that only makes sense in light of the conversation that produced it. It answers the
prompt rather than the reader: "equation 4 of the paper", "the failure this codebase has already
had once", "now with the pass-through removed". A week later nobody has the prompt, the paper has
been diverged from, and the sentence is either confusing or false.

Numbers from a run are the same thing. "Measured at 100k: whole records fall from 73.4% to 50.2%"
is one run, not a finding — five runs might say the opposite, and the rigorous version belongs in a
paper. A default may say what it is for; it may not claim what it is worth.

The code has to stand on its own. Say what the thing *is*, not what it was asked to be. If an
external source is genuinely worth naming, name it so the reader can reach it — a link and a
section, not "the paper". This applies to names and code as much as to comments.

Fields tell the story of the module in the order the reader meets them, which is usually the order
they are used: what reads the image, then what encodes it, then what writes the nodes down, then
what writes the relationships. Group what belongs together and separate the groups with a blank
line. Where a field must come earlier than its use — one field is built from another — that is the
exception, and the line that needs it should be the next one.

## Comments

Sparingly, so that a comment being there means it is worth reading.

- Only what the code and the names cannot express. If a comment restates the line below it, delete it.
- Short. A comment nobody finishes is a comment nobody reads.
- Never argue for the design against an alternative that is not in the code. Notes like "this rather
  than X, because X hedged" belong in a commit message or a report, not in the source — the reader
  has no X to compare against and the sentence will outlive the memory of it.
- Scaladoc when there is something interesting to say about a parameter or a return value. When
  there isn't, leave it out rather than restating the signature in prose.

## Structure

Prefer fewer functions. Every extra one is a context switch: the reader leaves the method, reads
something in isolation, and comes back holding it in their head.

- A step that only one method needs belongs *inside* that method, as a nested function.
- Often it should not be a function at all. A `val` with a good name says the same thing with less
  ceremony, and the name does the explaining.
- Top-level private helpers are for steps that several methods genuinely share.

## Dead code goes

We are looking for the right architecture, so ideas get tried and dropped. When one is dropped,
remove it — the switch that selects it, the branch that supports it, the comment that remembers it.

No backwards compatibility. If a new idea is better, the code should express *that* idea well rather
than carry both. Ablations come later, from the version history, not from options left lying around.

Comprehension beats configurability, which is a break from most public repositories. An option is
worth having when it means something in the model. An option that remembers an abandoned idea is
not, and it spreads: the pass-through term cost a flag, a branch in two losses, and a whole layer
of score wrappers around what the model actually answers — the reader paid for all of it on the way
to the part that runs.

## No mutable state

No `var`, no mutable collections. If one seems unavoidable, it is worth a second look — and if it
survives that, it is worth a comment saying why.

## DimWit and DeepWit

Follow the style of those repositories, core and examples both.

- `import dimwit.Conversions.given` and write scalars plainly. Not `Tensor0(vtype)(1f)`.
- `broadcastTo` only when nothing else will do. There is usually a broadcasting operator — `*!`,
  `+!`, `<=!`, `where_!` — and it reads better.
- Look for the nicest way the libraries offer before writing it by hand.
- If the function you want is missing, say so. Extending DimWit or DeepWit is the user's call, and a
  missing broadcast is worth reporting rather than working around.
