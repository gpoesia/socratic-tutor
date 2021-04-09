import React from 'react';

import T from './ternary-string.js';
import Add from './add.js';
import Arrow from './arrow.js';

const SuzzleInstructions = () => (
  <div>
    <h3>Instructions</h3>
    <p>Here is a brief reference of the rules of Suzzle.</p>
    <p><strong>Goal.</strong> The goal is to reach a sequence of symbols such that:
      (1) no two symbols have the same color, (2) the symbols are sorted by size,
      and (3) there are no circles. It is possible for the starting sequence to
      already satisfy these conditions.
    </p>
    <p><strong>Moves.</strong> You can perform any of the following 3 operations.</p>
    <p style={{ "margin-left": "2em" }}>
      <ulist>
        <li>Erase a circle.</li>
        <li>Swap the order of two adjacent symbols.</li>
        <li>
          Mix two adjacent symbols that have the same size. They will produce another two
          symbols, one having the same size and another having the very next size. These are
          all the possible combinations:
          <div className="problem">
            <T digits="#(a0)" /> <Add /> <T digits="#(a0)" /> <Arrow /> <T digits="#(a0 a1)" /> <br />
            <T digits="#(a0)" /> <Add /> <T digits="#(b0)" /> <Arrow /> <T digits="#(b0 a1)" /> <br />
            <T digits="#(a0)" /> <Add /> <T digits="#(c0)" /> <Arrow /> <T digits="#(c0 a1)" /> <br />
            <T digits="#(b0)" /> <Add /> <T digits="#(b0)" /> <Arrow /> <T digits="#(c0 a1)" /> <br />
            <T digits="#(b0)" /> <Add /> <T digits="#(c0)" /> <Arrow /> <T digits="#(a0 b1)" /> <br />
            <T digits="#(c0)" /> <Add /> <T digits="#(c0)" /> <Arrow /> <T digits="#(b0 b1)" /> <br />
          </div>
        </li>

      </ulist>
    </p>
  </div>
);
export default SuzzleInstructions;
