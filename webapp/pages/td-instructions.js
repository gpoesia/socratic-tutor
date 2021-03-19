import React, { useState, useEffect } from "react";
import dynamic from 'next/dynamic';
import useStore from '../lib/state';
import { ternaryDigitsEqual } from '../lib/ternary';
import { useRouter } from 'next/router';
import { apiRequest } from '../lib/api';
import _ from 'lodash';

import LinearProgress from '@material-ui/core/LinearProgress';
import { Button } from '@material-ui/core';

const T = dynamic(() => import('../lib/components/ternary-string.js'),
                  { ssr: false });
const Arrow = dynamic(() => import('../lib/components/arrow.js'),
                      { ssr: false });
const Add = dynamic(() => import('../lib/components/add.js'),
                    { ssr: false });
const SuzzleEditor = dynamic(() => import('../lib/components/suzzle-editor.js'),
                             { ssr: false });

const TernaryInstructions = () => {
  const router = useRouter();
  const next = () => router.push('/td-exercises');
  const [steps, setSteps] = useState([]);

  const canAdvance = steps.length > 0 && ternaryDigitsEqual(steps[steps.length - 1].join(''), "b3");

  return (
    <div className="content">
      <h1>Suzzle: a puzzle with symbols</h1>
      <p>
        In Suzzle, we are given a sequence of symbols, and our goal is to transform it
        until it is <em>irreducible</em>. Let's see what all that means.
      </p>
      <p>
        There are three kinds of symbols we can have:
        <T digits="#(a3 b3 c3)" />
      </p>
      <p>
        Each symbol can come in 6 different flavors:
        <T digits="#(a0 a1 a2 a3 a4 a5)" />
      </p>
      <p>
        The colors and sizes above always match: the largest symbol is always black,
        whereas the smaller sizes transition to red.
      </p>
      <p>
        In the puzzle, we'll be given a sequence of such symbols, with possibly mixed sizes and colors,
        and in any order. For example, let's take the following as an example:
      </p>
      <div className="problem">
        <T digits="#(c4 c3 b0 a3)" />
      </div>
      <p>
        Our goal is to transform these symbols until we meet three goal conditions:
      </p>
      <ol>
        <li>There should be at most one symbol of each color,</li>
        <li>The symbols should be ordered by size, from smallest to largest, and</li>
        <li>There should be no circles.</li>
      </ol>
      <p>
        In the case above, none of the conditions are met: there is one circle,
        the largest symbol is on the left, and both the triangle and the circle
        have the same color. How can we fix that?
      </p>
      <p>
        To make progress, there are also three moves we can make:
      </p>
      <ol>
        <li>We can erase a circle.</li>
        <li>You can swap adjacent symbols.</li>
        <li>
          If two adjacent symbols have <strong>the same size</strong>,
          we can "mix" them, which erases them and forms two new symbols.
        </li>
      </ol>
      <p>
        We'll see how to "mix" symbols in just a bit. For our last example, we won't need that.
        First, in that example, we can start by erasing the circle. This leaves us with:
      </p>
      <div className="problem">
        <T digits="#(c4 c3 b0 a3)" /> <Arrow />
        <T digits="#(c4 c3 b0)" />
      </div>
      <p>
        Now, we don't have circles anymore, and each symbol has a different color.
        The only problem is that the symbols are out of order, regarding their size.
        But we can fix this by swapping adjacent symbols a few times:
      </p>
      <div className="problem">
        <T digits="#(c4 c3 b0)" /> <Arrow />
        <T digits="#(c3 c4 b0)" /> <Arrow />
        <T digits="#(c3 b0 c4)" /> <Arrow />
        <T digits="#(b0 c3 c4)" /> <br/>
      </div>
      <p>
        And we're done! If you check the three goal conditions again, we now meet all three:
        each symbol has a different color, they are sorted by size and there are no circles.
      </p>
      <p>
        Finally, let's see how to "mix" symbols. We can only mix two symbols
        that are adjacent to each other, and that have the same color, but they can be
        different symbols. When we mix two symbols, we erase them both and get back two
        other symbols: one having the same size as the mixed symbols, and one that is
        of the next size. Here are all combinations we can get:
      </p>
      <div className="problem">
        <T digits="#(a3)" /> <Add /> <T digits="#(a3)" /> <Arrow /> <T digits="#(a3 a4)" /> <br />
        <T digits="#(a3)" /> <Add /> <T digits="#(b3)" /> <Arrow /> <T digits="#(b3 a4)" /> <br />
        <T digits="#(a3)" /> <Add /> <T digits="#(c3)" /> <Arrow /> <T digits="#(c3 a4)" /> <br />
        <T digits="#(b3)" /> <Add /> <T digits="#(b3)" /> <Arrow /> <T digits="#(c3 a4)" /> <br />
        <T digits="#(b3)" /> <Add /> <T digits="#(c3)" /> <Arrow /> <T digits="#(a3 b4)" /> <br />
        <T digits="#(c3)" /> <Add /> <T digits="#(c3)" /> <Arrow /> <T digits="#(b3 b4)" /> <br />
      </div>

      <p>The same combinations work if the symbols to be mixed are in the opposite order. For example,
        we could combine
        <T digits="#(c0)" /> <Add /> <T digits="#(b0)" /> to get <T digits="#(a0 b1)" />.
      </p>
      <p>
        Let's see another puzzle using mixing. Suppose we start with the folowing symbols:
      </p>
      <div className="problem">
        <T digits="#(c1 b2 b1)" />
      </div>
      <p>
        As it is, we can't combine anything. But we can if we first swap the two last symbols:
      </p>
      <div className="problem">
        <T digits="#(c1 b2 b1)" /> <Arrow /> <T digits="#(c1 b1 b2)" />
      </div>
      <p>
        Now, we can mix the two first symbols together. Looking back at the rules above, we see that we can do:
      </p>
      <div className="problem">
        <T digits="#(c1 b1 b2)" /> <Arrow /> <T digits="#(a1 c2 b2)" />
      </div>
      <p>
        Next, the last two symbols are of the same size. We can again mix those:
      </p>
      <div className="problem">
        <T digits="#(a1 c2 b2)" /> <Arrow /> <T digits="#(a1 a2 b3)" />
      </div>
      <p>
        We're almost done: now we just need to get rid of the circles:
      </p>
      <div className="problem">
        <T digits="#(a1 a2 b3)" /> <Arrow /> <T digits="#(a2 b3)" /> <Arrow /> <T digits="#(b3)" />
      </div>
      <p>
        Now that you have seen the rules, it's time to practice and learn some Suzzle!
        For that, you will go through a series of computer-generated exercises that will
        help you learn. You will first be asked to try to solve the exercise and submit an answer.
        If it is correct, you will advance to the next. Otherwise, the computer will show you
        its own answer step-by-step. To enter your answer, you will have an interface like the
        one below:
      </p>
      <SuzzleEditor
        problem={"c1 b2 b1"}
        steps={steps}
        setSteps={setSteps}
        onSubmit={() => {
          if (canAdvance)
            next();
          else
            alert("Please check your answer and try again. The correct solution was given in the text.");
        }}
      />
      <p>
        To "type" your solution, just click on symbols to add them, or the "backspace" button to erase
        the last symbol. You can add multiple steps to your answer. This is completely optional
        - we will only evaluate your last step. However, especially with longer puzzles, it may help
        you to make less mistakes. If the given Suzzle is already irreducible, simply type it again
        and submit.
      </p>
      <p>
        To advance to the next session, enter above the solution to the puzzle, and click "Submit last step".
        Note that this is exactly the puzzle we just explained step-by-step.
      </p>
    </div>
  );
};

export default TernaryInstructions;
