import React,{useState, useEffect} from "react";
import StepByStepSolution from "./step-by-step-solution";
import Checkbox from "@material-ui/core/Checkbox";

const RankSolutions = ({ solutions, checked, setChecked }) => {

    const [shuffledSolutions, setShuffledSolutions] = useState([]);

    useEffect(() => {
    let shuffledSolutions = solutions
      .map((a) => ({ sort: Math.random(), value: a }))
      .sort((a, b) => a.sort - b.sort)
      .map((a) => a.value);
      setShuffledSolutions(shuffledSolutions)
  }, [solutions]);

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
      {shuffledSolutions.map((solution) => (
        <div key={solution.id}>
          <StepByStepSolution solutionStrs={solution.steps} />
          <Checkbox
            checked={checked.has(solution.id)}
            onChange={() => handleChecked(solution.id)}
          />
        </div>
      ))}
    </div>
  );
};

export default RankSolutions;
