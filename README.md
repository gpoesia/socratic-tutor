# Socratic: Learning and Tutoring Symbolic Manipulation Domains

This project consists of (1) learning how to solve procedural educational domains,
and then (2) teaching people to do the same, using an expert domain model.

The educational domains themselves are implemented in Racket.
We're using [Algebraic Racket](https://docs.racket-lang.org/algebraic/ref.html),
since Algebraic Data Types are especially useful for recursive structures like
mathematical expressions. You can install all dependencies with:

```
raco pkg install --auto --pkgs algebraic brag date rebellion http-easy gregor
```

The learning algorithms are implemented in Python 3, using PyTorch. You can install
all Python dependencies using:

```
pip install -r requirements.txt
```

Finally, the human evaluation is done with a Web application, written using Next.js and React.
It is located under the `webapp` directory. To install all dependencies there, simply
use `npm install` on the `webapp` directory.

## Educational domains

Our first goal is to learn how to automatically solve exercises in a new educational domain.
Concretely, a domain defines:

(1) What exercises exist in the domain (e.g. what are algebraic equations),
(2) How to generate new exercises in that domain (e.g. an equations generator),
(3) Given one state, what are the actions available at that state (e.g. in an equation, could be changing the order of some terms, applying an operation to both sides, etc),
(4) Finally, how to detect that an exercise was solved (e.g. in equations, `x = 4` is solved, while `2x = 8` is still not)

This is roughly what you need to implement a new domain:

* First, we need to create new subtypes of terms to represent exercises in the new domain.
  This is done in `terms.rkt`. You need to add the appropriate constructors to the `Term` algebraic type,
  to the `Term?` predicate, and add a way of formatting these terms under the `format-term` function.
  Other functions in this file that handle each kind of term also need to be updated (such as `term-size` and
  `subterms`)
* Then, we need to augment our parser to recognize terms in the new domain.
  This mainly consists of changing the brag grammar in `grammar.rkt`, and then changing `term-parser.rkt`
  where needed.
* Now that we have a way of representing and parsing terms, we can proceed to (2), (3) and (4).
  Create a new file for the domain. You can copy `sorting.rkt` as a simple domain to start from.
  You'll need to define axioms for the new domain, then tactics, and finally the domain function,
  which applies all tactics and returns all possible next steps (3).
* In the same file, also implement a generator and a predicate that tells whether the exercise is solved.
* Finally, add your domain to `domains.rkt`, with the corresponding axioms. That's all, your domain is
  now fully integrated into the rest! We can train and evaluate various agents to learn it, and see how
  well they do.

## Learning agents

Several learning algorithms are implemented to learn the domains.
They are all in `agent.py`, which is a file that also implements evaluation.
More on this soon!