import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import useStore from "../lib/state";
import { useRouter } from "next/router";
import { apiRequest } from "../lib/api";
import _ from "lodash";
const TuringTestJson = require("./turing_test.json");

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
        Here is an example task that follows the same structure as the tasks you
        will see in the actual experiment.
      </p>
      <p>Consider this equation problem,</p>
      <p>{TuringTestJson["example"]["question"]}</p>
      <p>
        The following are 4 step-by-step solutions, of which two are written by human. All of them are correct.
        Please pick the two solutions that you believe are most likely to be written by
        human.
      </p>
      <RankSolutions
        solutions={TuringTestJson["example"]["solutions"]}
        checked={checked}
        setChecked={setChecked}
      />
      <p>
      To pick a solution, please check the box. You must pick exactly
        two solutions.
      </p>
      <p>To begin the experiment, please click on "Start".</p>
      <div className = "center-button">
      <button
        onClick={() => {
          if (canAdvance) next();
          else alert("You must select two solutions.");
        }}
      >
        Start
      </button>
      </div>
    </div>
  );
};

export default TuringTestInstructions;
