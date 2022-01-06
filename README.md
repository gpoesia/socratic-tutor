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

The environment that the agent interacts with is implemented in the backend of a localhost server. To start the server, run
```
racket environment.rkt
```
Wait until the message `Running environment server on 127.0.0.1:9898` to appear.

Now, we can perform training and evaluation, which is done by `agent.py`. Run the following command:
```
python agent.py [-h] --config CONFIG [--learn] [--experiment] [--eval] [--eval-checkpoints] [--debug] [--range RANGE] [--gpu GPU]
```
- `--config`: Path to configuration file, or inline JSON. A template configuration file for the `--learn` mode is given in `template_config.txt`. (Note: The template currently does not comprehensively list all possible configurations.)
- `--learn`: Put an agent to learn from the environment.
- `--experiment`: Run a batch of experiments with multiple agents and environments.
- `--eval`: Evaluate a learned policy.
- `eval-checkpoints`: Show the evolution of a learned policy during interaction.
- `debug`: Enable debug messages.
- `range RANGE`: Range of experiments to run. Format: 2-5 means range [2, 5).Used to split experiments across multiple machines. Default: all.
- `gpu GPU`: Which GPU to use (e.g. `"cuda:0"`); defaults to CPU if none is specified.

## Rust environments

We're testing a faster implementation of the environments in Rust. Right now, the Racket environments
are the gold standard, and only the equations domain has an implementation in Rust. However, to train
larger models, we'll likely need the Rust environments, since they can be more than 100x faster.
If you want to try this new setup, follow the steps below:

* First, install a recent Rust compiler (1.50+). The easiest way to do it is with rust-up. Simply
  visit [https://rustup.rs/](https://rustup.rs/) and follow the instructions there. To check your
  installation, run `rustc --version` in the command line: you should get something like
  `rustc 1.51.0 (2fd73fabe 2021-03-23)`.
* Then, compile the dynamic library:

```
$ cd /path/to/socratic-tutor/commoncore
$ cargo build --release
```

  This might take a few minutes. It should download all dependencies and then compile our library.
  If all goes well, you should find the library at `target/release/libcommoncore.so`
  (or the equivalent extension in your operating system).
* Finally, we only need to place that library in a location that we can import from Python.
  Simply create a symbolic link at the root of the project and named `commoncore.so`, that points
  to the compiled library:

```
user@machine:~/socratic-tutor$ ln -s commoncore/target/release/libcommoncore.so ./commoncore.so
```

And that's it! If you open a Python shell, you should be able to directly `import commoncore`.
Also, now that you set up the symlink, if you compile the Rust library again (e.g. after it was updated),
we'll automatically pick up the latest version from Python.