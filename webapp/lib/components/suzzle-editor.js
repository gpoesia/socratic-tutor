import React from "react";
import { Button } from '@material-ui/core';
import T from "./ternary-string";
import _ from 'lodash';
import IconButton from '@material-ui/core/IconButton';
import Backspace from '@material-ui/icons/Backspace';

const EMPTY = "<empty>";

const SuzzleEditor = ({ problem, steps, setSteps, onSubmit }) => {
  const addSymbol = (s) => {
    if (steps.length == 0) {
      return;
    }
    const keptSteps = steps.slice(0, steps.length - 1);
    const newLastStep = steps[steps.length - 1].concat([s]);
    setSteps(keptSteps.concat([newLastStep]));
  };

  const backspace = (s) => {
    const keptSteps = steps.slice(0, steps.length - 1);
    const lastStep = steps[steps.length - 1];
    const newLastStep = lastStep.slice(0, lastStep.length - 1);
    setSteps(keptSteps.concat([newLastStep]));
  };

  const addStep = () => setSteps(steps.concat([[]]));
  const removeStep = () => setSteps(steps.slice(0, steps.length - 1));

  return (
    <div className="problem">
      <div className="challenge">
        <label>Starting suzzle:</label>
        <T digits={problem} />
      </div>
      <div>
        Your solution:
        <ol>
          {
            steps.map((s, i) => (
              <li key={i}>
                { s.length ? <T digits={s.join(" ")} /> : EMPTY }
                { s.length && (i + 1) == steps.length
                  ? <IconButton color="primary" onClick={backspace}><Backspace /></IconButton>
                  : null
                }
              </li>
            ))
          }
        </ol>
        <span className="action">Add:</span>
        {
          ['a', 'b', 'c'].map((d, j) =>
            _.range(6).map((p, i) =>
              <span key={"d" + i + j} onClick={() => addSymbol(d + p)}>
                <T digits={d + p} />
              </span>
            )
          )
        }

      </div>
      <span className="action">
        <Button variant="contained" color="primary" onClick={addStep}>New step</Button>
      </span>
      <span className="action">
        <Button disabled={steps.length == 0} variant="contained"
                color="primary" onClick={removeStep}>
          Remove last step
        </Button>
      </span>
      <span className="action">
        <Button disabled={steps.length == 0} variant="contained"
                color="primary" onClick={onSubmit}>
          Submit last step
        </Button>
      </span>
    </div>
  );
}

export default SuzzleEditor;
