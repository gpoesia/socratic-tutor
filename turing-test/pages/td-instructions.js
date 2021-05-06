import React, { useState, useEffect } from "react";
import dynamic from 'next/dynamic';
import {Row} from "react"
import useStore from '../lib/state';
import { useRouter } from 'next/router';
import { apiRequest } from '../lib/api';
import _ from 'lodash';
const Example = require('./example0.json');

const RankSolutions = dynamic(() => import('../lib/components/rank-solutions.js'),
                  { ssr: false });


const TuringTestInstructions = () => {
  const router = useRouter();
  const next = () => router.push('/td-exercises');
  const [steps, setSteps] = useState([]);

  const canAdvance = steps.length > 0 && ternaryDigitsEqual(steps[steps.length - 1].join(''), "b3");

  return (
    <div className="content">
      <h1>Human or Machine: Step-by-Step Solutions for Equation Problems</h1>
      <p>
        Here is an example task that follows the same structure as the ones you will see in the actual experiment.
      </p>
      <p>
        Consider this equation problem,
      </p>
      <p>{Example["question"]}</p>
      <p>The following are 4 solutions, of which two are written by human. Please rank them according to your confidence in which one is written by a human.
        Your rank 1 choice should be the solution you are most confident about.
      </p>
      <RankSolutions solutions = {Example["solutions"]} />
      <p>
        To rank the solutions, just click on symbols to add them, or the "backspace" button to erase
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

export default TuringTestInstructions;
