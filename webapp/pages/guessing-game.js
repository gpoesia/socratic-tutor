import React, { useState, useEffect, useRef } from 'react';
import CircularProgress from '@material-ui/core/CircularProgress';
import IconButton from '@material-ui/core/IconButton';
import Backspace from '@material-ui/icons/Backspace';
import styles from '../styles/guessing-game.module.css';
import shuffle from 'lodash.shuffle';
import { MathComponent as Tex } from 'mathjax-react';

export default function GuessingGame(props) {
  const [trace, setTrace] = useState(null);
  const [problem, setProblem] = useState(null);
  const [lastProblem, setLastProblem] = useState(null);
  const [mistakes, setMistakes] = useState([]);
  const [nextStep, setNextStep] = useState(1);
  const [permutation, setPermutation] = useState([]);
  const [optionChosen, setOptionChosen] = useState(false);

  useEffect(async () => {
    if (problem === null) {
      const nextProblem = await tutorRequest('next-problem',
                                             {
                                               policy: props.policy,
                                               lastProblem: lastProblem && lastProblem.id,
                                               mistakes,
                                             });

      setProblem(nextProblem);
      setMistakes([]);
      setNextStep(1);
      setPermutation(nextProblem.negatives.map(_ => Math.random() < 0.5));
    }
  });

  if (problem === null) {
    return (
      <div className={styles.gameContainer}>
        <CircularProgress />
      </div>
    );
  }

  const makeAttempt = (attempt) => {
    const correct = attempt === problem.solution[nextStep];

    if (!correct) {
      setMistakes(mistakes.concat([nextStep]));
    }

    setOptionChosen(true);
    setTimeout(() => {
      setOptionChosen(false);
      setNextStep(nextStep + 1);
    }, correct ? 500 : 2000);
  };

  const nextProblem = () => {
    setLastProblem(problem);
    setProblem(null);
  };

  const negatives = problem.negatives.map(l => l[l.length - 1]);
  const finished = nextStep > negatives.length;

  const options = !finished && [
    <button
      key={0}
      disabled={optionChosen}
      className={optionChosen ? styles.correctOption : styles.option}
      onClick={() => makeAttempt(problem.solution[nextStep])}
    >
      <Tex tex={ problem.solution[nextStep]} />
    </button>,
    <button
      key={1}
      disabled={optionChosen}
      className={optionChosen ? styles.incorrectOption : styles.option}
      onClick={() => makeAttempt(negatives[nextStep - 1])}
    >
      <Tex tex={negatives[nextStep - 1]} />
    </button>
  ];

  if (!finished && permutation[nextStep]) {
    options.reverse();
  }

  return (
    <div className={styles.gameContainer}>
      <h1>Guess the next step that the AI took!</h1>
      <div className={styles.stepsList}>
        {
          problem.solution.slice(0, nextStep).map((step, i) => (
            <span key={i}><Tex display={false} tex={"\\mbox{" + (i+1) + ".}\\enspace " + step} /></span>
          ))
        }
        { !finished
          ? <div key={0}>
              <p>What should be the next step?</p>
              <div className={styles.options}>
                { options }
              </div>
            </div>
          : <div key={1}>
              <p>You scored {problem.solution.length - mistakes.length}/{problem.solution.length}</p>
              <button onClick={() => nextProblem()}>Next problem</button>
            </div>
        }
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
