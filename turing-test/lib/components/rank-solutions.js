import React from "react";
import StepByStepSolution from "./step-by-step-solution";
import Checkbox from "@material-ui/core/Checkbox";

const RankSolutions = ({ solutions, checked, setChecked }) => {
  const handleChecked = (idx) => {
    if (checked.has(idx)) {
      let newChecked = new Set(checked);
      newChecked.delete(idx);
      setChecked(newChecked);
    } else if (checked.size < 2) {
      let newChecked = new Set(checked);
      newChecked.add(idx);
      setChecked(newChecked);
    }
  };
  return (
    <div className="solutions_root">
      {solutions.map((solution, idx) => (
        <div key={idx}>
          <StepByStepSolution solutionStrs={solution} />
          <Checkbox
            checked={checked.has(idx)}
            onChange={() => handleChecked(idx)}
          />
        </div>
      ))}
    </div>
  );
};

export default RankSolutions;
