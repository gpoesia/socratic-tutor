import React, { useState, useEffect, useRef } from 'react';
import CircularProgress from '@material-ui/core/CircularProgress';
import IconButton from '@material-ui/core/IconButton';
import Backspace from '@material-ui/icons/Backspace';
import styles from '../../styles/guessing-game.module.css';
import lodash from 'lodash';
import { MathComponent as Tex } from 'mathjax-react';

export default function GuessingGame(props) {
  const {
    problem,
    permutation,
    optionChosen,
    makeAttempt
  } = props;

  if (problem === undefined) {
    return (
      <div className={styles.gameContainer}>
        <CircularProgress />
      </div>
    );
  }

  const exercisePrompt = (problem.type === 'guess-step'
                          ? <div>
                              <p>If you are solving the equation:</p>
                              <Tex tex={problem.exercise['state']} />
                              <p>What would be your next step?</p>
                            </div>
                          : <div>
                              <p>
                                In which of the equations below would your next step
                                be <em>"{problem.exercise['step-description']}"</em>?
                              </p>
                            </div>);

  const optionsText = (problem.type === 'guess-step'
                       ? [
                         problem.exercise['pos']['step-description'],
                         problem.exercise['neg']['step-description'],
                         problem.exercise['distractors'][0]['step-description'],
                       ]
                       : [
                         <Tex tex={problem.exercise['correct']} />,
                         <Tex tex={problem.exercise['distractors'][0]} />,
                         <Tex tex={problem.exercise['distractors'][1]} />,
                       ]);

  const optionsB = [
    <button
      key={0}
      disabled={optionChosen}
      className={optionChosen ? styles.correctOption : styles.option}
      onClick={() => makeAttempt(0)}
    >
      {optionsText[0]}
    </button>,
    <button
      key={1}
      disabled={optionChosen}
      className={optionChosen ? styles.incorrectOption : styles.option}
      onClick={() => makeAttempt(1)}
    >
      {optionsText[1]}
    </button>,
    <button
      key={2}
      disabled={optionChosen}
      className={optionChosen ? styles.incorrectOption : styles.option}
      onClick={() => makeAttempt(2)}
    >
      {optionsText[2]}
    </button>
  ];

  const options = lodash.range(3).map(i => optionsB[permutation[i]]);

  return (
    <div className={styles.gameContainer}>
      { exercisePrompt }
      <div className={styles.stepsList}>
        <div className={styles.options}>
          { options }
        </div>
      </div>
    </div>
  );
}
