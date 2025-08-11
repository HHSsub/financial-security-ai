"""
Dashboard integration module for Financial Security AI Model
"""

import os
import sys
import json
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add financial_security_ai to path
sys.path.append('/workspace/financial_security_ai')

# Import local modules
from data_analysis import analyze_data
from inference import FinancialSecurityModelInference

class FinancialSecurityDashboard:
    def __init__(self, model_path=None):
        """
        Initialize the dashboard integration
        
        Args:
            model_path: Path to the model to use (default to best model)
        """
        self.model_path = model_path or "upstage/SOLAR-10.7B-Instruct-v1.0"
        self.inference_model = None
        self.test_data = None
        self.results_data = None
        self.analysis_results = None
        
        # Create directories
        os.makedirs("/workspace/dashboard/public/data", exist_ok=True)
        os.makedirs("/workspace/dashboard/public/assets/images", exist_ok=True)
    
    def load_data(self, test_path="/workspace/uploads/test.csv", results_path=None):
        """
        Load and analyze test data
        
        Args:
            test_path: Path to test data
            results_path: Path to results data (if available)
        """
        # Load test data
        self.test_data = pd.read_csv(test_path)
        
        # Load results data if available
        if results_path and os.path.exists(results_path):
            self.results_data = pd.read_csv(results_path)
        
        # Analyze data
        self.analysis_results = analyze_data(test_path)
        
        # Prepare data for dashboard
        self._prepare_dashboard_data()
    
    def _prepare_dashboard_data(self):
        """Prepare data for dashboard display"""
        # Create summary data
        summary_data = {
            "total_questions": len(self.test_data),
            "multiple_choice": self.analysis_results["question_types"]["multiple_choice"],
            "subjective": self.analysis_results["question_types"]["subjective"],
            "avg_question_length": self.analysis_results["length_stats"]["mean_length"],
            "model_used": self.model_path.split("/")[-1],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save summary data to JSON
        with open("/workspace/dashboard/public/data/summary_data.json", "w") as f:
            json.dump(summary_data, f)
        
        # Convert test data to JSON for dashboard
        test_sample = self.test_data.head(10).to_dict(orient="records")
        with open("/workspace/dashboard/public/data/test_sample.json", "w") as f:
            json.dump(test_sample, f)
        
        # If we have results, prepare results data
        if self.results_data is not None:
            results_sample = self.results_data.head(10).to_dict(orient="records")
            with open("/workspace/dashboard/public/data/results_sample.json", "w") as f:
                json.dump(results_sample, f)
    
    def load_inference_model(self):
        """Load the inference model"""
        print(f"Loading inference model: {self.model_path}")
        self.inference_model = FinancialSecurityModelInference(
            model_path=self.model_path,
            load_in_4bit=True
        )
        print("Model loaded successfully")
    
    def run_inference_sample(self, num_samples=5):
        """
        Run inference on a sample of test data
        
        Args:
            num_samples: Number of samples to run inference on
            
        Returns:
            DataFrame with inference results
        """
        if self.inference_model is None:
            self.load_inference_model()
        
        # Sample test data
        if num_samples >= len(self.test_data):
            sample_df = self.test_data
        else:
            sample_df = self.test_data.sample(num_samples)
        
        # Initialize results
        results = []
        
        # Run inference on each sample
        for _, row in sample_df.iterrows():
            question_id = row['ID']
            question = row['Question']
            
            # Generate answer
            start_time = time.time()
            answer = self.inference_model.generate_response(question)
            inference_time = time.time() - start_time
            
            # Store result
            results.append({
                'ID': question_id,
                'Question': question,
                'Answer': answer,
                'InferenceTime': inference_time
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save results to JSON
        results_json = results_df.to_dict(orient="records")
        with open("/workspace/dashboard/public/data/inference_results.json", "w") as f:
            json.dump(results_json, f)
        
        return results_df
    
    def generate_visualizations(self):
        """Generate visualizations for dashboard"""
        # 1. Question type distribution pie chart
        plt.figure(figsize=(8, 8))
        question_types = [
            self.analysis_results["question_types"]["multiple_choice"],
            self.analysis_results["question_types"]["subjective"]
        ]
        plt.pie(question_types, labels=['Multiple Choice', 'Subjective'],
                autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#2196F3'])
        plt.title('Question Type Distribution')
        plt.savefig('/workspace/dashboard/public/assets/images/question_type_dist.png')
        
        # 2. Question length distribution
        plt.figure(figsize=(10, 6))
        self.test_data['question_length'] = self.test_data['Question'].apply(len)
        sns.histplot(self.test_data['question_length'], bins=20, kde=True)
        plt.title('Question Length Distribution')
        plt.xlabel('Length (characters)')
        plt.ylabel('Count')
        plt.savefig('/workspace/dashboard/public/assets/images/question_length_dist.png')
        
        # 3. If we have inference results, plot inference times
        if hasattr(self, 'inference_results') and self.inference_results is not None:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.inference_results['InferenceTime'], bins=20, kde=True)
            plt.title('Inference Time Distribution')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Count')
            plt.savefig('/workspace/dashboard/public/assets/images/inference_time_dist.png')
    
    def update_dashboard_components(self):
        """Update dashboard components with AI model data"""
        # Update Dashboard.jsx component
        with open("/workspace/dashboard/src/components/Dashboard.jsx", "r") as f:
            dashboard_jsx = f.read()
        
        # Modify dashboard component to include our data
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
    // Load data from JSON files
    fetch('/data/summary_data.json')
      .then(response => response.json())
      .then(data => setSummaryData(data))
      .catch(error => console.error('Error loading summary data:', error));
      
    fetch('/data/test_sample.json')
      .then(response => response.json())
      .then(data => setTestSample(data))
      .catch(error => console.error('Error loading test sample:', error));
      
    fetch('/data/inference_results.json')
      .then(response => response.json())
      .then(data => setInferenceResults(data))
      .catch(error => console.error('Error loading inference results:', error));
  }, []);

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold mb-4">금융보안 특화 AI 모델 대시보드</h1>
        <p className="text-gray-600 mb-4">
          FSKU 평가지표에 대한 금융보안 특화 AI 모델의 성능을 분석하는 대시보드입니다.
          이 대시보드에서는 데이터 분석, 모델 성능, 추론 결과를 시각화하여 보여줍니다.
        </p>
        <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-4">
          <p className="text-sm">
            <span className="font-bold">현재 모델:</span> {summaryData?.model_used || '로딩 중...'} <br />
            <span className="font-bold">마지막 업데이트:</span> {summaryData?.timestamp || '로딩 중...'}
          </p>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatsCard
          title="총 문항 수"
          value={summaryData?.total_questions || '로딩 중...'}
          trend="neutral"
          icon="📊"
        />
        <StatsCard
          title="객관식 문항"
          value={summaryData?.multiple_choice || '로딩 중...'}
          trend="up"
          icon="📝"
        />
        <StatsCard
          title="주관식 문항"
          value={summaryData?.subjective || '로딩 중...'}
          trend="up"
          icon="📑"
        />
        <StatsCard
          title="평균 문항 길이"
          value={summaryData?.avg_question_length ? Math.round(summaryData.avg_question_length) + ' 자' : '로딩 중...'}
          trend="neutral"
          icon="📏"
        />
      </div>

      {/* Charts */}
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
              <BarChart customData={inferenceResults.map((item, index) => ({
                name: `Q${index+1}`,
                value: parseFloat(item.InferenceTime).toFixed(2)
              }))} />
            ) : (
              <div className="flex items-center justify-center h-full">
                <p>추론 데이터 로딩 중...</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Test Sample Table */}
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
              {testSample.map((item, index) => (
                <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                  <td className="py-2 px-4 border-b">{item.ID}</td>
                  <td className="py-2 px-4 border-b">
                    <div className="whitespace-pre-wrap">{item.Question}</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Inference Results */}
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
                {inferenceResults.map((item, index) => (
                  <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                    <td className="py-2 px-4 border-b">{item.ID}</td>
                    <td className="py-2 px-4 border-b">
                      <div className="whitespace-pre-wrap line-clamp-2">{item.Question}</div>
                    </td>
                    <td className="py-2 px-4 border-b">
                      <div className="whitespace-pre-wrap line-clamp-2">{item.Answer}</div>
                    </td>
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
        
        # Write updated dashboard component
        with open("/workspace/dashboard/src/components/Dashboard.jsx", "w") as f:
            f.write(updated_dashboard)
        
        # Update Header.jsx to reflect our application
        with open("/workspace/dashboard/src/components/Header.jsx", "r") as f:
            header_jsx = f.read()
        
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
        
        # Write updated header component
        with open("/workspace/dashboard/src/components/Header.jsx", "w") as f:
            f.write(updated_header)
        
        # Update index.html title
        with open("/workspace/dashboard/index.html", "r") as f:
            index_html = f.read()
        
        updated_index_html = index_html.replace("<title>Analytics Dashboard</title>", "<title>금융보안 특화 AI 모델 대시보드</title>")
        
        with open("/workspace/dashboard/index.html", "w") as f:
            f.write(updated_index_html)