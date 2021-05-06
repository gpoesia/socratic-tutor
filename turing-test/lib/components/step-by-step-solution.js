import React from "react";
import { colors } from "just-mix";


const StepByStepSolution = ({ solutionStrs }) => {
  return (
    <div className="StepByStepSolution">
      {solutionStrs.map((step, idx) => (
        <div key={idx}> {step} </div>
      ))}
    </div>
  );
};

export default StepByStepSolution;
