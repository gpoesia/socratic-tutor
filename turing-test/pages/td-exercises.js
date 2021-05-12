import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import useStore from "../lib/state";
import { useRouter } from "next/router";
import { apiRequest } from "../lib/api";
import _ from "lodash";
import CircularProgress from "@material-ui/core/CircularProgress";
import LinearProgress from "@material-ui/core/LinearProgress";

const TestQuestionsJson = require("./turing_test.json");

const RankSolutions = dynamic(
  () => import("../lib/components/rank-solutions.js"),
  { ssr: false }
);

const TuringTest = () => {
  const router = useRouter();
  const sessionId = useStore((state) => state.id);

  const [TestQuestions, setTestQuestions] = useState(null);
  const [checked, setChecked] = useState(new Set());
  const [currQIdx, setQIdx] = useState(0);

  useEffect(() => {
    //load questions from Json file. Shuffle questions.
    if (TestQuestions === null) {
      let questions = TestQuestionsJson["experiment"]
        .map((a) => ({ sort: Math.random(), value: a }))
        .sort((a, b) => a.sort - b.sort)
        .map((a) => a.value);
      setTestQuestions(questions);
    }
  });

  const canAdvance = checked.size == 2;
  const canGoBack = currQIdx - 1 >= 0;
  const next = async () => {
    //save answer for current problem
    await apiRequest("save-answers", {
      id: sessionId,
      qStr: TestQuestions[currQIdx]["question"],
      qAns: Array.from(checked),
    });
    //advance to the next problem/end of experiment
    if (currQIdx + 1 == TestQuestions.length) {
      router.push("/end");
    } else {
      setQIdx(currQIdx + 1);
      setChecked(new Set());
    }
  };

  const back = () => {
    setQIdx(currQIdx - 1);
    setChecked(new Set());
  };

  if (TestQuestions === null) {
    return (
      <div className="content">
        <CircularProgress />
      </div>
    );
  }

  return (
    <div className="content">
      <LinearProgress
        variant="determinate"
        value={((currQIdx + 1) / TestQuestions.length) * 100}
      />

      <h1>Human or Machine: Step-by-Step Solutions for Equation Problems</h1>
      <h2>Progress: {(currQIdx + 1)} / {TestQuestions.length}</h2>
      <p>Consider this equation problem,</p>
      <br/>
      <p className="center-button">{TestQuestions[currQIdx]["question"]}</p>
      <p>
        The following are 4 step-by-step solutions, of which two are written by
        human. Please pick the two solutions that you believe are most likely to
        be written by human.
      </p>
      <RankSolutions
        solutions={TestQuestions[currQIdx]["solutions"]}
        checked={checked}
        setChecked={setChecked}
      />
      <p>
        To pick a solution, please check the box. You must pick exactly two
        solutions.
      </p>
      <div className="left-right-buttons">
        <button disabled={!canGoBack} onClick={back}>
          Go Back
        </button>
        <button disabled={!canAdvance} onClick={next}>
          Continue
        </button>
      </div>
    </div>
  );
};

export default TuringTest;
