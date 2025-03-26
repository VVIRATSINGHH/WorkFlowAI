from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import os
import requests
import json
import time
from typing import Dict, List, Any
import hashlib
import unittest
from unittest import mock
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Text, DateTime, Float, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set your credentials
os.environ["GOOGLE_GEMINI_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["DATABASE_URL"] = "sqlite:///./test.db"  # Default to SQLite for simplicity

Base = declarative_base()

class AgentVersionRecord(Base):
    _tablename_ = 'agent_versions'
    id = Column(String, primary_key=True)
    agent_name = Column(String)
    version = Column(String)
    update_date = Column(DateTime)
    changelog = Column(Text)

class ProjectStateRecord(Base):
    _tablename_ = 'project_states'
    project_id = Column(String, primary_key=True)
    status = Column(String)
    current_phase = Column(String)
    tasks = Column(JSON)
    dependencies = Column(JSON)
    agents_involved = Column(JSON)
    last_updated = Column(DateTime)

class PerformanceMetricRecord(Base):
    _tablename_ = 'performance_metrics'
    id = Column(String, primary_key=True)
    agent_name = Column(String)
    timestamp = Column(DateTime)
    response_time = Column(Float)
    accuracy = Column(Float)
    relevance = Column(Float)
    task_completion = Column(Float)

class AgentVersion(BaseModel):
    version: str
    update_date: str
    changelog: str

class PerformanceMetrics(BaseModel):
    response_time: List[float]
    accuracy: List[float]
    relevance: List[float]
    task_completion: List[bool]

class ProjectState(BaseModel):
    project_id: str
    status: str  # planning, execution, monitoring, completed
    current_phase: str
    tasks: Dict[str, Any]
    dependencies: Dict[str, List[str]]
    agents_involved: Dict[str, str]
    last_updated: datetime

class Task(BaseModel):
    task_id: str
    description: str
    status: str  # not_started, in_progress, completed, blocked
    assigned_agent: str
    dependencies: List[str]
    artifacts: List[str]
    logs: List[str]

class AgentMessage(BaseModel):
    sender: str
    receiver: str
    message_id: str
    content: str
    timestamp: datetime
    status: str  # sent, delivered, read, responded

class EnhancedGeminiAssistantAgent(AssistantAgent):
    def _init_(self, name: str, description: str, system_message: str = None):
        super()._init_(name=name, description=description)
        self.conversation_history = []
        self.performance_metrics = PerformanceMetrics(
            response_time=[],
            accuracy=[],
            relevance=[],
            task_completion=[]
        )
        self.version_history = [AgentVersion(version="1.0.0", update_date=str(datetime.now()), changelog="Initial version")]
        self.system_message = system_message or f"You are {name}, {description}"
        self.rag_enabled = False
        self.vectorstore = None
        self.max_context_length = 2048  # Adjust based on Gemini's actual context window
        self.message_broker = None
        self.persistence_manager = None

    def initialize_rag(self, documents: List[str], validate_quality: bool = True):
        """Initialize Retrieval Augmented Generation with a vector store"""
        if validate_quality:
            quality_prompt = f"Act as a document quality assessor. Evaluate if these documents are suitable for RAG initialization:\n{documents}\nProvide your evaluation as JSON with 'valid' (true/false) and 'reason'."
            evaluation = self.query_gemini(quality_prompt)
            try:
                eval_result = json.loads(evaluation)
                if not eval_result.get("valid", False):
                    raise ValueError(f"Document quality check failed: {eval_result.get('reason', 'No reason provided')}")
            except:
                raise ValueError("Document validation failed")
        
        self.vectorstore = FAISS.from_texts(documents, OpenAIEmbeddings())
        self.rag_enabled = True

    def query_gemini(self, prompt: str) -> str:
        """Query Google Gemini API with optional RAG context"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['GOOGLE_GEMINI_API_KEY']}"
        }
        
        pruned_history = self._prune_conversation_history(prompt)
        
        messages = [{"author": "system", "content": self.system_message}]
        messages.extend(pruned_history)
        messages.append({"author": "user", "content": prompt})
        
        data = {
            "prompt": {"messages": messages},
            "temperature": 0.2,
            "candidate_count": 1,
            "max_output_tokens": 1024
        }
        
        if self.rag_enabled and self.vectorstore:
            relevant_docs = self.vectorstore.similarity_search(prompt, k=3)
            data["prompt"]["messages"].insert(-1, {"author": "system", "content": f"Relevant context: {relevant_docs}"})
        
        start_time = time.time()
        try:
            response = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateMessage",
                headers=headers,
                data=json.dumps(data)
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise HTTPException(status_code=500, detail="API request failed")
        
        response_time = time.time() - start_time
        self.performance_metrics.response_time.append(response_time)
        
        try:
            response_content = response.json()["candidates"][0]["content"]
        except (KeyError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing response: {e}")
            raise HTTPException(status_code=500, detail="Error parsing response")
        
        self.conversation_history.append({"author": "user", "content": prompt})
        self.conversation_history.append({"author": "assistant", "content": response_content})
        
        if self.persistence_manager:
            self.persistence_manager.save_performance_metric(
                self.name,
                response_time,
                1.0,  # Placeholder for accuracy
                1.0,  # Placeholder for relevance
                True   # Placeholder for task completion
            )
        
        return response_content

    def _prune_conversation_history(self, new_prompt: str) -> List[Dict[str, str]]:
        """Prune conversation history to prevent context window overflow"""
        total_length = len(new_prompt)
        pruned_history = []
        
        for msg in reversed(self.conversation_history):
            total_length += len(msg["content"])
            if total_length > self.max_context_length:
                break
            pruned_history.append(msg)
        
        return list(reversed(pruned_history))

    def generate_response(self, message: str) -> str:
        """Generate a response with validation"""
        if not isinstance(message, str):
            raise ValueError("Input message must be a string")
        
        response = self.query_gemini(message)
        self._validate_response(response)
        self._track_performance(message, response)
        return response

    def _validate_response(self, response: str) -> None:
        """Validate response quality"""
        if not response or len(response.strip()) < 10:
            raise ValueError("Response is empty or too short")
        if len(response) > 2000:
            raise ValueError("Response is too long")

    def _track_performance(self, message: str, response: str) -> None:
        """Track performance metrics"""
        # Implement actual accuracy and relevance tracking in production
        self.performance_metrics.accuracy.append(1.0)  # Placeholder
        self.performance_metrics.relevance.append(1.0)  # Placeholder
        self.performance_metrics.task_completion.append(True)  # Placeholder

    def evaluate_performance(self) -> Dict[str, float]:
        """Evaluate agent performance based on metrics"""
        if len(self.performance_metrics.response_time) == 0:
            return {
                "response_time": 0.0,
                "accuracy": 0.0,
                "relevance": 0.0,
                "task_completion": 0.0
            }
        
        avg_response_time = sum(self.performance_metrics.response_time) / len(self.performance_metrics.response_time)
        avg_accuracy = sum(self.performance_metrics.accuracy) / len(self.performance_metrics.accuracy)
        avg_relevance = sum(self.performance_metrics.relevance) / len(self.performance_metrics.relevance)
        task_completion_rate = sum(self.performance_metrics.task_completion) / len(self.performance_metrics.task_completion)
        
        return {
            "response_time": avg_response_time,
            "accuracy": avg_accuracy,
            "relevance": avg_relevance,
            "task_completion": task_completion_rate
        }

    def retrain(self, training_data: List[Dict[str, str]], tuning_parameters: Dict[str, Any] = None) -> None:
        """Retrain the agent with new data"""
        current_version = self.version_history[-1].version
        version_parts = current_version.split(".")
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version = ".".join(version_parts)
        
        print(f"Retraining agent {self.name} to version {new_version} with {len(training_data)} samples")
        
        if self.persistence_manager:
            self.persistence_manager.save_agent_version(
                self.name,
                new_version,
                datetime.now(),
                f"Retrained with {len(training_data)} samples"
            )
        
        self.version_history.append(
            AgentVersion(
                version=new_version,
                update_date=str(datetime.now()),
                changelog=f"Retrained with {len(training_data)} samples"
            )
        )

    def set_message_broker(self, broker):
        self.message_broker = broker
        broker.subscribe(self.name, self)

    def send_message(self, receiver: str, content: str):
        if not self.message_broker:
            raise ValueError("Message broker not initialized")
            
        return self.message_broker.send_message(self.name, receiver, content)
        
    def receive_message(self, message: AgentMessage):
        print(f"Received message from {message.sender}: {message.content}")
        self.process_incoming_message(message)
        
    def process_incoming_message(self, message: AgentMessage):
        # Custom processing logic for each agent type
        pass

    def set_persistence_manager(self, persistence_manager):
        self.persistence_manager = persistence_manager

class QualityControlAgent(EnhancedGeminiAssistantAgent):
    def _init_(self):
        super()._init_(
            name="QualityControl",
            description="Validates output quality of other agents.",
            system_message="You are a quality assurance expert. Your job is to evaluate the outputs of other AI agents for accuracy, completeness, and professional standards."
        )

    def validate_output(self, agent_name: str, output: str) -> bool:
        """Validate output from another agent"""
        validation_prompt = f"""Evaluate if the following output from agent {agent_name} is complete, accurate, and meets professional standards:

Output:
{output}

Provide your evaluation as JSON with the following structure:
{{
  "valid": true|false,
  "reason": "Reason for validation result",
  "score": number between 0 and 100
}}
"""
        
        try:
            response = self.generate_response(validation_prompt)
            validation_result = json.loads(response)
            
            if not isinstance(validation_result, dict):
                raise ValueError("Invalid validation result format")
            
            if "valid" not in validation_result or not isinstance(validation_result["valid"], bool):
                raise ValueError("Missing or invalid 'valid' field")
            
            if "reason" not in validation_result or not isinstance(validation_result["reason"], str):
                raise ValueError("Missing or invalid 'reason' field")
            
            if "score" not in validation_result or not isinstance(validation_result["score"], (int, float)):
                raise ValueError("Missing or invalid 'score' field")
            
            return validation_result["valid"]
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

class MessageBroker:
    def _init_(self):
        self.message_queue = []
        self.message_history = []
        self.subscribers = {}

    def send_message(self, sender: str, receiver: str, content: str):
        message = AgentMessage(
            sender=sender,
            receiver=receiver,
            message_id=str(uuid.uuid4()),
            content=content,
            timestamp=datetime.now(),
            status="sent"
        )
        self.message_queue.append(message)
        return message
        
    def deliver_messages(self):
        while self.message_queue:
            message = self.message_queue.pop(0)
            message.status = "delivered"
            self.message_history.append(message)
            
            if message.receiver in self.subscribers:
                for agent in self.subscribers[message.receiver]:
                    agent.receive_message(message)
                
    def subscribe(self, agent_name: str, agent):
        if agent_name not in self.subscribers:
            self.subscribers[agent_name] = []
        self.subscribers[agent_name].append(agent)

class DataPersistenceManager:
    def _init_(self, db_connection_string: str = None):
        if db_connection_string is None:
            db_connection_string = os.environ.get("DATABASE_URL", "sqlite:///./test.db")
        self.engine = create_engine(db_connection_string)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def save_agent_version(self, agent_name: str, version: str, update_date: datetime, changelog: str):
        session = self.Session()
        try:
            version_record = AgentVersionRecord(
                id=str(uuid.uuid4()),
                agent_name=agent_name,
                version=version,
                update_date=update_date,
                changelog=changelog
            )
            session.add(version_record)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
            
    def save_project_state(self, project_state: ProjectState):
        session = self.Session()
        try:
            project_record = ProjectStateRecord(
                project_id=project_state.project_id,
                status=project_state.status,
                current_phase=project_state.current_phase,
                tasks=json.dumps(project_state.tasks),
                dependencies=json.dumps(project_state.dependencies),
                agents_involved=json.dumps(project_state.agents_involved),
                last_updated=project_state.last_updated
            )
            session.add(project_record)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()
            
    def load_project_state(self, project_id: str) -> ProjectState:
        session = self.Session()
        try:
            record = session.query(ProjectStateRecord).filter_by(project_id=project_id).first()
            if not record:
                raise ValueError(f"Project {project_id} not found in database")
                
            return ProjectState(
                project_id=record.project_id,
                status=record.status,
                current_phase=record.current_phase,
                tasks=json.loads(record.tasks),
                dependencies=json.loads(record.dependencies),
                agents_involved=json.loads(record.agents_involved),
                last_updated=record.last_updated
            )
        except:
            raise
        finally:
            session.close()
            
    def save_performance_metric(self, agent_name: str, response_time: float, accuracy: float, relevance: float, task_completion: bool):
        session = self.Session()
        try:
            metric_record = PerformanceMetricRecord(
                id=str(uuid.uuid4()),
                agent_name=agent_name,
                timestamp=datetime.now(),
                response_time=response_time,
                accuracy=accuracy,
                relevance=relevance,
                task_completion=1.0 if task_completion else 0.0
            )
            session.add(metric_record)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

class ProjectManagerSystem:
    def _init_(self):
        self.agents = {}
        self.quality_control = QualityControlAgent()
        self.active_projects = {}
        self.message_broker = MessageBroker()
        self.persistence_manager = DataPersistenceManager()
        
        # Initialize agents
        self.quality_control.set_message_broker(self.message_broker)
        self.quality_control.set_persistence_manager(self.persistence_manager)
        
    def create_agent(self, name: str, description: str, system_message: str = None) -> EnhancedGeminiAssistantAgent:
        """Create a new agent with version control"""
        if name in self.agents:
            raise ValueError(f"Agent with name {name} already exists")
        
        agent = EnhancedGeminiAssistantAgent(name, description, system_message)
        agent.set_message_broker(self.message_broker)
        agent.set_persistence_manager(self.persistence_manager)
        self.agents[name] = agent
        return agent

    def get_agent(self, name: str) -> EnhancedGeminiAssistantAgent:
        """Get an existing agent"""
        if name not in self.agents:
            raise ValueError(f"Agent with name {name} does not exist")
        return self.agents[name]

    def validate_agent_output(self, agent_name: str, output: str) -> bool:
        """Validate output from an agent"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent with name {agent_name} does not exist")
        
        return self.quality_control.validate_output(agent_name, output)

    def retrain_agent(self, agent_name: str, training_data: List[Dict[str, str]], tuning_parameters: Dict[str, Any] = None) -> None:
        """Retrain an agent with new data"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent with name {agent_name} does not exist")
        
        agent = self.agents[agent_name]
        agent.retrain(training_data, tuning_parameters)
        logger.info(f"Agent {agent_name} retrained to version {agent.version_history[-1].version}")

    def create_project(self, project_id: str, requirements: str) -> None:
        """Create a new project with appropriate agents"""
        if project_id in self.active_projects:
            raise ValueError(f"Project with ID {project_id} already exists")
        
        # Analyze requirements and determine needed agents
        analysis_prompt = f"""Analyze the following project requirements and determine what types of AI agents are needed:

Requirements:
{requirements}

Provide your response as JSON with the following structure:
{{
  "agents": [
    {{
      "name": "Agent name",
      "description": "Agent description",
      "system_message": "System message for the agent"
    }}
  ]
}}
"""
        
        # Use the quality control agent to analyze requirements
        try:
            analysis_result = self.quality_control.generate_response(analysis_prompt)
            analysis_data = json.loads(analysis_result)
            
            if not isinstance(analysis_data, dict) or "agents" not in analysis_data:
                raise ValueError("Invalid analysis format")
            
            agents_to_create = analysis_data["agents"]
            project_agents = {}
            
            for agent_config in agents_to_create:
                agent = self.create_agent(
                    name=agent_config["name"],
                    description=agent_config["description"],
                    system_message=agent_config.get("system_message")
                )
                project_agents[agent_config["name"]] = agent
            
            project_state = ProjectState(
                project_id=project_id,
                status="planning",
                current_phase="initialization",
                tasks={},
                dependencies={},
                agents_involved={agent_name: agent.description for agent_name, agent in project_agents.items()},
                last_updated=datetime.now()
            )
            
            self.active_projects[project_id] = {
                "requirements": requirements,
                "agents": project_agents,
                "state": project_state
            }
            
            self.persistence_manager.save_project_state(project_state)
            
        except Exception as e:
            logger.error(f"Project creation failed: {e}")
            raise

    def execute_project(self, project_id: str) -> None:
        """Execute a project by initiating agent workflows"""
        if project_id not in self.active_projects:
            raise ValueError(f"Project with ID {project_id} does not exist")
        
        project = self.active_projects[project_id]
        project_state = project["state"]
        project_agents = project["agents"]
        
        # Implement project execution logic here
        # This would typically involve coordinating between agents
        # and monitoring their progress
        
        logger.info(f"Executing project {project_id} with agents: {list(project_agents.keys())}")
        
        # Example: Send initial message to project planner
        planner = next((agent for agent_name, agent in project_agents.items() if "planner" in agent_name.lower()), None)
        if planner:
            planner.send_message("QualityControl", "Begin project planning phase")
            self.message_broker.deliver_messages()
        
        # Update project state
        project_state.status = "execution"
        project_state.current_phase = "planning"
        project_state.last_updated = datetime.now()
        self.persistence_manager.save_project_state(project_state)

    def update_project_state(self, project_id: str, new_status: str, new_phase: str):
        """Update the state of a project"""
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found")
            
        project = self.active_projects[project_id]
        project["state"].status = new_status
        project["state"].current_phase = new_phase
        project["state"].last_updated = datetime.now()
        
        self.persistence_manager.save_project_state(project["state"])

class MonitoringDashboard:
    def _init_(self, project_manager: ProjectManagerSystem):
        self.project_manager = project_manager
        self.app = FastAPI()
        
        @self.app.get("/agents")
        def get_agents():
            return {"agents": list(self.project_manager.agents.keys())}
        
        @self.app.get("/agent/{agent_name}/performance")
        def get_agent_performance(agent_name: str):
            if agent_name not in self.project_manager.agents:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            agent = self.project_manager.agents[agent_name]
            performance = agent.evaluate_performance()
            return {
                "agent_name": agent_name,
                "performance": performance,
                "version_history": [v.dict() for v in agent.version_history]
            }
        
        @self.app.get("/projects")
        def get_projects():
            return {"projects": list(self.active_projects.keys())}
        
        @self.app.get("/project/{project_id}/status")
        def get_project_status(project_id: str):
            if project_id not in self.project_manager.active_projects:
                raise HTTPException(status_code=404, detail="Project not found")
            
            project = self.project_manager.active_projects[project_id]
            return {
                "project_id": project_id,
                "status": project["state"].status,
                "current_phase": project["state"].current_phase,
                "agents_involved": project["state"].agents_involved,
                "last_updated": project["state"].last_updated
            }

class TestAgentFunctionality(unittest.TestCase):
    def setUp(self):
        self.persistence_manager = DataPersistenceManager("sqlite:///./test.db")
        self.message_broker = MessageBroker()
        self.agent = EnhancedGeminiAssistantAgent(
            name="TestAgent",
            description="An agent for testing purposes",
            system_message="You are a test agent for validation purposes"
        )
        self.agent.set_message_broker(self.message_broker)
        self.agent.set_persistence_manager(self.persistence_manager)
        
    @mock.patch('requests.post')
    def test_query_gemini_success(self, mock_post):
        mock_response = mock.Mock()
        mock_response.json.return_value = {"candidates": [{"content": "Test response"}]}
        mock_post.return_value = mock_response
        
        response = self.agent.query_gemini("Test message")
        self.assertEqual(response, "Test response")
        
    def test_message_sending(self):
        receiver = EnhancedGeminiAssistantAgent(
            name="ReceiverAgent",
            description="An agent that receives messages",
            system_message="You receive messages from other agents"
        )
        receiver.set_message_broker(self.message_broker)
        
        message = self.agent.send_message("ReceiverAgent", "Hello, this is a test message")
        self.message_broker.deliver_messages()
        
        # Check if message was received
        # Implementation would depend on how receive_message is tracked
        self.assertTrue(len(self.message_broker.message_history) > 0)

class AgentVersionController:
    def _init_(self, git_repo_url: str, deployment_environment: str):
        self.git_repo_url = git_repo_url
        self.deployment_environment = deployment_environment
        self.current_version = None
        
    def clone_repository(self):
        # Implement git repository cloning
        pass
        
    def checkout_branch(self, branch_name: str):
        # Implement git branch checkout
        pass
        
    def pull_latest_changes(self):
        # Implement git pull
        pass
        
    def run_tests(self):
        # Implement test execution
        pass
        
    def build_agent(self):
        # Implement agent building process
        pass
        
    def deploy_agent(self):
        # Implement deployment logic
        pass
        
    def rollback_to_previous_version(self):
        # Implement rollback logic
        pass
        
    def promote_version(self, version: str, environment: str):
        # Implement version promotion between environments
        pass
        
    class VersioningSystem:
        def __init__(self):
            self.current_version = None  # Initialize necessary attributes

        def clone_repository(self):
            print("Cloning repository...")

        def checkout_branch(self, branch_name):
            print(f"Checking out branch: {branch_name}")

        def pull_latest_changes(self):
            print("Pulling latest changes...")

        def run_tests(self):
            print("Running tests...")

        def build_agent(self):
            print("Building agent...")

        def deploy_agent(self):
            print("Deploying agent...")

        def rollback_to_previous_version(self):
            print("Rolling back to previous version...")

        def versioning_pipeline(self, new_version: str):
            try:
                self.clone_repository()
                self.checkout_branch(f"versions/{new_version}")  # This line should now work
                self.pull_latest_changes()
                self.run_tests()
                self.build_agent()
                self.deploy_agent()
                self.current_version = new_version
                return True
            except Exception as e:
                self.rollback_to_previous_version()
                print(f"Deployment failed: {e}")
                return False


