import React, { useState, useEffect } from "react";
import dynamic from 'next/dynamic';
import useStore from '../lib/state';
import { useRouter } from 'next/router';
import { apiRequest } from '../lib/api';
import _ from 'lodash';
import LinearProgress from '@material-ui/core/LinearProgress';

const GuessingGame = dynamic(() => import('../lib/components/guessing-game.js'),
                             { ssr: false });

const Exercises = () => {
  const id = useStore(state => state.id);
  const router = useRouter();
  const [problem, setProblem] = useState(undefined);
  const [permutation, setPermutation] = useState([]);
  const [optionChosen, setOptionChosen] = useState(false);
  const [progress, setProgress] = useState(0);

  // Fetch a new problem or move to the post-test.
  useEffect(async () => {
    if (problem === undefined) {
      const { problem, progress, done } = await apiRequest('next-problem', { id });

      if (done) {
        router.push('/assessment?stage=post');
      } else {
        setProblem(problem);
        setOptionChosen(false);
        setPermutation(_.shuffle(_.range(3)));
        setProgress(progress);
      }
    }
  });

  const makeAttempt = async (response) => {
    const correct = (response === 0);
    setOptionChosen(true);

    await apiRequest('save-answers',
                     { id,
                       stage: 'exercises',
                       answers: [{ id: problem.id, response }],
                     });

    setTimeout(() => {
      setOptionChosen(false);
      setProblem(undefined);
    }, correct ? 500 : 5000);
  };

  return (
    <div className="content">
      <h1>Exercises</h1>
      <LinearProgress value={progress * 100} variant="determinate" />
      <GuessingGame
        problem={problem}
        permutation={permutation}
        optionChosen={optionChosen}
        makeAttempt={makeAttempt}
      />
    </div>
  );
};
export default Exercises;
