# Socratic Tutor

Implementation of the Socratic Tutor project.

For now, the short-term goal is to get the solver to generate the step-by-step
solution to simple math problems, so that we can think about/play with asking questions.

The solver is implemented in Racket.
We're using [Algebraic Racket](https://docs.racket-lang.org/algebraic/ref.html),
since Algebraic Data Types are especially useful for recursive structures like
mathematical expressions. You can install it with:

```
raco pkg install --auto algebraic
```

## Prototype

By running `racket tutor-example.rkt`, you'll interact with a preliminary version
of the tutor. It still does not ask questions, or finds solutions.
Rather, it verifies whether each step in the student's solution is correct.
Here's an example dialogue:

```
$ racket tutor-example.rkt
Let's solve a math problem! Given:
(1): ((1 + 2) + 3) = (-9x + 10x)
You need to meet:
(G1): x = ??
>>> 7 = x
Hmm, I could not verify that. Try again?
>>> 6 = x
OK! Let's add that to what we know:
(2): 6 = x
>>> x = 6
OK! Let's add that to what we know:
(3): x = 6
Great, this matches the goal x = ??
You're done!
```

## Current stage and next steps

These are some clear short-term goals:

- [x] Have a representation for terms (`x + 2y`) and equalities between terms (`x + 2y = x - 1`)
- [x] Implement term simplification using random search (`x + y + x + y` --> `2 * (x + y)`).
- [ ] Implement simple tactics for solving a problem (e.g. "Isolate x in the equation", "Substitute an equation in another", "Simplify both sides").
- [ ] Implement step-by-step solver for simple systems of linear equations.

After that, we'll likely want to (not necessarily in this order):

- [ ] Think more concretely about interaction, especially how to generate questions and answers.
- [ ] Try to make the solver work with simple problems from actual math exercise websites/textbooks.
- [ ] Pick other categories of problems to expand what we can solve.
