import React, { useState, useEffect, useRef } from 'react';
import CircularProgress from '@material-ui/core/CircularProgress';
import IconButton from '@material-ui/core/IconButton';
import Backspace from '@material-ui/icons/Backspace';
import styles from '../styles/guessing-game.module.css';
import lodash from 'lodash';
import { MathComponent as Tex } from 'mathjax-react';

export default function GuessingGame(props) {
  const [problem, setProblem] = useState(null);
  const [lastExercise, setLastExercise] = useState(null);
  const [success, setSuccess] = useState(false);
  const [permutation, setPermutation] = useState([]);
  const [optionChosen, setOptionChosen] = useState(false);

  useEffect(async () => {
    if (problem === null) {
      const nextProblem = await tutorRequest('next-problem',
                                             {
                                               policy: props.policy,
                                               lastExercise: lastExercise && lastExercise.id,
                                               succeded: success,
                                             });

      console.log('Got problem:', nextProblem);

      setProblem(nextProblem);
      setOptionChosen(false);
      setPermutation(lodash.shuffle(lodash.range(3)));
    }
  });

  if (problem === null) {
    return (
      <div className={styles.gameContainer}>
        <CircularProgress />
      </div>
    );
  }

  const nextProblem = () => {
    setLastExercise(problem);
    setProblem(null);
  };

  const makeAttempt = (correct) => {
    setSuccess(correct);
    setOptionChosen(true);
    setTimeout(() => {
      setOptionChosen(false);
      nextProblem();
    }, correct ? 500 : 3000);
  };

  const optionsP = [
    <button
      key={0}
      disabled={optionChosen}
      className={optionChosen ? styles.correctOption : styles.option}
      onClick={() => makeAttempt(true)}
    >
      <Tex tex={ problem.pos.step } />
    </button>,
    <button
      key={1}
      disabled={optionChosen}
      className={optionChosen ? styles.incorrectOption : styles.option}
      onClick={() => makeAttempt(false)}
    >
      <Tex tex={ problem.neg.step } />
    </button>,
    <button
      key={2}
      disabled={optionChosen}
      className={optionChosen ? styles.incorrectOption : styles.option}
      onClick={() => makeAttempt(false)}
    >
      <Tex tex={ problem.distractors[0].step } />
    </button>
  ];

  const options = lodash.range(3).map(i => optionsP[permutation[i]]);

  return (
    <div className={styles.gameContainer}>
      <h1>Guess the next step that the AI took!</h1>
      <div className={styles.stepsList}>
        <span><Tex display={false} tex={"\\mbox{1. }\\enspace " + problem.state } /></span>
        <p>What should be the next step?</p>
        <div className={styles.options}>
          { options }
        </div>
      </div>
    </div>
  );
}

async function tutorRequest(endpoint, parameters) {
  const req = await fetch('/api/' + endpoint + '?params=' + encodeURIComponent(JSON.stringify(parameters)));
  console.log('Fetched API:', req);
  const res = await req.json();
  return res;
}
