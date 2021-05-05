import React, { useState, useEffect } from "react";
import dynamic from 'next/dynamic';
import useStore from '../lib/state';
import { useRouter } from 'next/router';
import { apiRequest } from '../lib/api';
import { extractTernaryDigits, ternaryDigitsEqual } from '../lib/ternary';
import _ from 'lodash';

import LinearProgress from '@material-ui/core/LinearProgress';
import CircularProgress from '@material-ui/core/CircularProgress';

import { Button } from '@material-ui/core';

const SuzzleEditor = dynamic(() => import('../lib/components/suzzle-editor.js'),
                             { ssr: false });
const SuzzleInstructions = dynamic(() => import('../lib/components/suzzle-instructions.js'),
                                   { ssr: false });


const TernaryPostTest = () => {
  const sessionId = useStore(state => state.id);
  const router = useRouter();
  const [problems, setProblems] = useState(null);
  const [nextProblem, setNextProblem] = useState(0);
  const [steps, setSteps] = useState([[]]);

  useEffect(async () => {
    if (problems === null) {
      const postTestProblems = await apiRequest('post-test', { id: sessionId });
      setProblems(postTestProblems);
    } else if (nextProblem === problems.length) {
      router.push('/end');
    }
  });

  if (problems === null || nextProblem === problems.length) {
    return (
      <div className="content">
        <CircularProgress />
      </div>
    );
  }

  const problem = problems[nextProblem];

  const onSubmit = async () => {
    const correct = ternaryDigitsEqual(
      steps[steps.length - 1].join(''),
      problem.solution,
    );

    await apiRequest('save-answers',
                     {
                       id: sessionId,
                       stage: "assessment-post",
                       answers: [{ "id": problem.id, steps, correct }],
                     });

    setSteps([[]]);
    setNextProblem(nextProblem + 1);
  };

  console.log('problems:', problems);
  console.log('nextProblem:', nextProblem);
  console.log('progress:', 100*(nextProblem / problems.length));

  return (
    <div className="content">
      <h1>The Suzzle Challenge :: Final Test</h1>

      <LinearProgress variant="determinate" value={100*(nextProblem / problems.length)} />

      <p>To test your understanding, let's solve the final challenges.</p>
      <div className="challenge-root">
        <div className="challenge-content">
          <SuzzleEditor
            problem={problem.problem}
            steps={steps}
            setSteps={setSteps}
            onSubmit={onSubmit}
          />
        </div>
        <div className="challenge-instructions">
          <SuzzleInstructions />
        </div>
      </div>

    </div>
  );
};

export default TernaryPostTest;
