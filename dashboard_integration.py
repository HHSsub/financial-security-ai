# dashboard_integration.py
"""
Dashboard integration module for Financial Security AI Model
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Path setup: import local modules from this project root ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from data_analysis import analyze_data
from inference import FinancialSecurityModelInference


# ---- helpers ----
def to_builtin(o):
    """Recursively convert numpy types to Python builtins so json.dump works."""
    if isinstance(o, dict):
        return {k: to_builtin(v) for k, v in o.items()}
    if isinstance(o, list):
        return [to_builtin(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


class FinancialSecurityDashboard:
    def __init__(self, model_path=None, load_in_4bit=True):
        """
        Initialize the dashboard integration

        Args:
            model_path: HuggingFace model id or local path
            load_in_4bit: use bitsandbytes 4-bit quant (Colab/Linux 권장)
        """
        self.model_path = model_path or "upstage/SOLAR-10.7B-Instruct-v1.0"
        self.load_in_4bit = load_in_4bit

        self.inference_model = None
        self.test_data: pd.DataFrame | None = None
        self.results_data: pd.DataFrame | None = None
        self.analysis_results = None
        self.inference_results: pd.DataFrame | None = None

        # I/O dirs (create if not exists)
        self.data_dir = "/workspace/dashboard/public/data"
        self.assets_dir = "/workspace/dashboard/public/assets/images"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.assets_dir, exist_ok=True)

        # convenience paths
        self.summary_path = os.path.join(self.data_dir, "summary_data.json")
        self.test_sample_path = os.path.join(self.data_dir, "test_sample.json")
        self.results_sample_path = os.path.join(self.data_dir, "results_sample.json")
        self.infer_results_path = os.path.join(self.data_dir, "inference_results.json")

    # ------------------------------------------------------------------ #
    # Data loading / preprocessing                                       #
    # ------------------------------------------------------------------ #
    def load_data(self, test_path="/workspace/uploads/test.csv", results_path=None):
        """
        Load and analyze test data

        Args:
            test_path: Path to test.csv
            results_path: Optional path to existing results csv
        """
        # Load test data
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"❌ test file not found: {test_path}")
        self.test_data = pd.read_csv(test_path)

        # Optional previous results
        if results_path and os.path.exists(results_path):
            self.results_data = pd.read_csv(results_path)

        # Analyze (returns dict of stats)
        self.analysis_results = analyze_data(test_path)

        # Prepare artifacts for dashboard
        self._prepare_dashboard_data()

    def _prepare_dashboard_data(self):
        """Prepare json artifacts used by the web dashboard."""
        assert self.test_data is not None, "Call load_data() first."

        qtypes = self.analysis_results.get("question_types", {})
        lengths = self.analysis_results.get("length_stats", {})

        summary_data = {
            "total_questions": len(self.test_data),
            "multiple_choice": qtypes.get("multiple_choice", 0),
            "subjective": qtypes.get("subjective", 0),
            "avg_question_length": lengths.get("mean_length", 0),
            "model_used": self.model_path.split("/")[-1],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Save summary
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(to_builtin(summary_data), f, ensure_ascii=False, indent=2)

        # Save small sample of test rows
        test_sample = self.test_data.head(10).to_dict(orient="records")
        with open(self.test_sample_path, "w", encoding="utf-8") as f:
            json.dump(to_builtin(test_sample), f, ensure_ascii=False, indent=2)

        # If we have results, save a sample
        if self.results_data is not None:
            results_sample = self.results_data.head(10).to_dict(orient="records")
            with open(self.results_sample_path, "w", encoding="utf-8") as f:
                json.dump(to_builtin(results_sample), f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------ #
    # Inference                                                          #
    # ------------------------------------------------------------------ #
    def load_inference_model(self):
        """Instantiate the HF model for inference."""
        print(f"Loading inference model: {self.model_path} (4bit={self.load_in_4bit})")
        self.inference_model = FinancialSecurityModelInference(
            model_path=self.model_path,
            load_in_4bit=self.load_in_4bit,
        )
        print("✅ Model loaded")

    def run_inference_sample(self, num_samples=5) -> pd.DataFrame:
        """
        Run inference on a sample (or the full set if num_samples >= len)

        Returns:
            DataFrame with columns: ID, Question, Answer, InferenceTime
        """
        if self.inference_model is None:
            self.load_inference_model()

        assert self.test_data is not None, "Call load_data() first."

        # Pick subset deterministically for reproducibility
        if num_samples >= len(self.test_data):
            sample_df = self.test_data.copy()
        else:
            sample_df = self.test_data.sample(num_samples, random_state=42)

        results = []
        for _, row in sample_df.iterrows():
            qid = row["ID"]
            question = row["Question"]

            start = time.time()
            answer = self.inference_model.generate_response(question)
            dt = time.time() - start

            results.append(
                {
                    "ID": int(qid) if isinstance(qid, (np.integer,)) else qid,
                    "Question": question,
                    "Answer": answer,
                    "InferenceTime": float(dt),
                }
            )

        results_df = pd.DataFrame(results)
        self.inference_results = results_df

        # Persist for dashboard
        with open(self.infer_results_path, "w", encoding="utf-8") as f:
            json.dump(to_builtin(results_df.to_dict(orient="records")), f, ensure_ascii=False, indent=2)

        return results_df

    # ------------------------------------------------------------------ #
    # Visualization artifacts (optional)                                 #
    # ------------------------------------------------------------------ #
    def generate_visualizations(self):
        """Generate and save static charts under assets/images."""
        assert self.test_data is not None, "Call load_data() first."

        # 1) Question type distribution
        plt.figure(figsize=(8, 8))
        q_mc = int(self.analysis_results["question_types"].get("multiple_choice", 0))
        q_sub = int(self.analysis_results["question_types"].get("subjective", 0))
        plt.pie(
            [q_mc, q_sub],
            labels=["Multiple Choice", "Subjective"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4CAF50", "#2196F3"],
        )
        plt.title("Question Type Distribution")
        plt.savefig(os.path.join(self.assets_dir, "question_type_dist.png"))
        plt.close()

        # 2) Question length distribution
        plt.figure(figsize=(10, 6))
        tmp = self.test_data.copy()
        tmp["question_length"] = tmp["Question"].apply(len)
        sns.histplot(tmp["question_length"], bins=20, kde=True)
        plt.title("Question Length Distribution")
        plt.xlabel("Length (characters)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(self.assets_dir, "question_length_dist.png"))
        plt.close()

        # 3) Inference times (if any)
        if self.inference_results is not None and len(self.inference_results) > 0:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.inference_results["InferenceTime"], bins=20, kde=True)
            plt.title("Inference Time Distribution")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Count")
            plt.savefig(os.path.join(self.assets_dir, "inference_time_dist.png"))
            plt.close()

    # ------------------------------------------------------------------ #
    # Front-end source patchers (optional, safe-guarded)                 #
    # ------------------------------------------------------------------ #
    def update_dashboard_components(self):
        """Update dashboard React components if they exist."""
        # Dashboard.jsx
        dash_path = "/workspace/dashboard/src/components/Dashboard.jsx"
        if os.path.exists(dash_path):
            updated_dashboard = """import React, { useState, useEffect } from 'react';
import StatsCard from './StatsCard';
import LineChart from './charts/LineChart';
import AreaChart from './charts/AreaChart';
import BarChart from './charts/BarChart';
import PieChart from './charts/PieChart';
import RadarChart from './charts/RadarChart';
import GaugeChart from './charts/GaugeChart';
import TreeMap from './charts/TreeMap';
import BubbleChart from './charts/BubbleChart';

function Dashboard() {
  const [summaryData, setSummaryData] = useState(null);
  const [testSample, setTestSample] = useState([]);
  const [inferenceResults, setInferenceResults] = useState([]);

  useEffect(() => {
    fetch('/data/summary_data.json').then(r => r.json()).then(setSummaryData).catch(console.error);
    fetch('/data/test_sample.json').then(r => r.json()).then(setTestSample).catch(console.error);
    fetch('/data/inference_results.json').then(r => r.json()).then(setInferenceResults).catch(console.error);
  }, []);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-4">금융보안 특화 AI 모델 대시보드</h1>
        <p className="text-gray-600 mb-4">
          FSKU 평가지표에 대한 금융보안 특화 AI 모델의 성능을 분석하는 대시보드입니다.
        </p>
        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-4">
          <p className="text-sm">
            <span className="font-bold">현재 모델:</span> {summaryData?.model_used || '로딩 중...'} <br />
            <span className="font-bold">마지막 업데이트:</span> {summaryData?.timestamp || '로딩 중...'}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatsCard title="총 문항 수" value={summaryData?.total_questions || '...'} trend="neutral" icon="📊" />
        <StatsCard title="객관식 문항" value={summaryData?.multiple_choice || '...'} trend="up" icon="📝" />
        <StatsCard title="주관식 문항" value={summaryData?.subjective || '...'} trend="up" icon="📑" />
        <StatsCard title="평균 문항 길이" value={summaryData?.avg_question_length ? Math.round(summaryData.avg_question_length)+' 자' : '...'} trend="neutral" icon="📏" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-bold mb-4">문항 유형 분포</h2>
          <div className="h-64">
            <PieChart customData={summaryData ? [
              { name: '객관식', value: summaryData.multiple_choice },
              { name: '주관식', value: summaryData.subjective }
            ] : []} />
          </div>
        </div>

        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-bold mb-4">모델 추론 시간 분포</h2>
          <div className="h-64">
            {inferenceResults.length > 0 ? (
              <BarChart customData={inferenceResults.map((item, idx) => ({
                name: `Q${idx+1}`,
                value: parseFloat(item.InferenceTime).toFixed(2)
              }))} />
            ) : (
              <div className="flex items-center justify-center h-full"><p>추론 데이터 로딩 중...</p></div>
            )}
          </div>
        </div>
      </div>

      <div className="bg-white p-4 rounded-lg shadow mb-8">
        <h2 className="text-lg font-bold mb-4">테스트 데이터 샘플</h2>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white">
            <thead>
              <tr className="bg-gray-100 text-gray-700 text-left">
                <th className="py-2 px-4 border-b">ID</th>
                <th className="py-2 px-4 border-b">질문</th>
              </tr>
            </thead>
            <tbody className="text-gray-600">
              {testSample.map((item, i) => (
                <tr key={i} className={i % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                  <td className="py-2 px-4 border-b">{item.ID}</td>
                  <td className="py-2 px-4 border-b"><div className="whitespace-pre-wrap">{item.Question}</div></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {inferenceResults.length > 0 && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h2 className="text-lg font-bold mb-4">모델 추론 결과</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white">
              <thead>
                <tr className="bg-gray-100 text-gray-700 text-left">
                  <th className="py-2 px-4 border-b">ID</th>
                  <th className="py-2 px-4 border-b">질문</th>
                  <th className="py-2 px-4 border-b">답변</th>
                  <th className="py-2 px-4 border-b">추론시간(초)</th>
                </tr>
              </thead>
              <tbody className="text-gray-600">
                {inferenceResults.map((item, i) => (
                  <tr key={i} className={i % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                    <td className="py-2 px-4 border-b">{item.ID}</td>
                    <td className="py-2 px-4 border-b"><div className="whitespace-pre-wrap line-clamp-2">{item.Question}</div></td>
                    <td className="py-2 px-4 border-b"><div className="whitespace-pre-wrap line-clamp-2">{item.Answer}</div></td>
                    <td className="py-2 px-4 border-b">{parseFloat(item.InferenceTime).toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
export default Dashboard;"""
            with open(dash_path, "w", encoding="utf-8") as f:
                f.write(updated_dashboard)

        # Header.jsx
        header_path = "/workspace/dashboard/src/components/Header.jsx"
        if os.path.exists(header_path):
            updated_header = """import React from 'react';
function Header() {
  return (
    <header className="bg-indigo-700 text-white px-6 py-4 shadow-md">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
          </svg>
          <div>
            <h1 className="text-xl font-bold">금융보안 특화 AI 모델</h1>
            <p className="text-xs opacity-75">FSKU 평가지표 기반 분석 대시보드</p>
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="hidden md:block">
            <div className="flex items-center bg-indigo-800 rounded-full px-3 py-1">
              <div className="h-2 w-2 rounded-full bg-green-400 mr-2"></div>
              <span className="text-xs font-medium">모델 상태: 정상</span>
            </div>
          </div>
          <button className="text-sm bg-indigo-600 hover:bg-indigo-500 py-1 px-3 rounded transition duration-300">
            모델 실행
          </button>
        </div>
      </div>
    </header>
  );
}
export default Header;"""
            with open(header_path, "w", encoding="utf-8") as f:
                f.write(updated_header)

        # index.html title
        index_path = "/workspace/dashboard/index.html"
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                idx = f.read()
            idx = idx.replace("<title>Analytics Dashboard</title>", "<title>금융보안 특화 AI 모델 대시보드</title>")
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(idx)
