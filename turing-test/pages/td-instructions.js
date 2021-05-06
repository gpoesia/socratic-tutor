import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import useStore from "../lib/state";
import { useRouter } from "next/router";
import { apiRequest } from "../lib/api";
import { Button } from '@material-ui/core';
import _ from "lodash";
const Example = require("./example0.json");

const RankSolutions = dynamic(
  () => import("../lib/components/rank-solutions.js"),
  { ssr: false }
);

const TuringTestInstructions = () => {
  const router = useRouter();
  const next = () => router.push("/td-exercises");
  const [checked, setChecked] = useState(new Set());
  const canAdvance = checked.size == 2;

  return (
    <div className="content">
      <h1>Human or Machine: Step-by-Step Solutions for Equation Problems</h1>
      <p>
        Here is an example task that follows the same structure as the ones you
        will see in the actual experiment.
      </p>
      <p>Consider this equation problem,</p>
      <p>{Example["question"]}</p>
      <p>
        The following are 4 solutions, of which two are written by human. Please
        pick the two solutions that you believe are most likely to be written by
        human. [Please rank them according to your confidence in which one is
        written by a human. Your rank 1 choice should be the solution you are
        most confident about.]
      </p>
      <RankSolutions
        solutions={Example["solutions"]}
        checked={checked}
        setChecked={setChecked}
      />
      <p>
        To pick a solution, just click on the checkbox. You must pick exactly
        two solutions.
      </p>
      <p>To begin the experiment, please click on "Start".</p>
      <div className = "center-button">
      <Button

        onClick={() => {
          if (canAdvance) next();
          else alert("You must select two solutions.");
        }}
      >
        Start
      </Button>
      </div>
    </div>
  );
};

export default TuringTestInstructions;
