from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(
    title="Tourism AI Agent API",
    description="API that provides information about tourism places using CrewAI agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TourismRequest(BaseModel):
    prompt: str

def generate_tourism_place(prompt: str) -> str:
    llm = LLM(
        model="gemini/gemini-2.0-flash",
        temperature=0.1
    )

    place_tell = Agent(
        role="Help people to know about tourism places",
        goal="Provide information about popular tourist destinations",
        verbose=True,
        backstory="You are a knowledgeable travel guide and have information about various tourist places so you help people to know more about these places",
        llm=llm
    )

    agent_task = Task(
        description=f"""Take the following prompt and generate a list of various tourism places that match with user's input.
        Use the latest data available on the internet to provide accurate information.
        Ensure that the output is well structured and easy to read by the user. dont use any extra stars or hyphens.
        Prompt:
        '''{prompt}'''""",
        agent=place_tell,
        expected_output="List of various tourism places with a brief description of each place",
    )

    crew = Crew(
        agents=[place_tell],
        tasks=[agent_task],
        verbose=True
    )

    result = crew.kickoff()
    return str(result.tasks_output[0].raw)

@app.post("/generate-tourism")
async def get_tourism_places(request: TourismRequest) -> Dict[str, str]:
    try:
        result = generate_tourism_place(request.prompt)
        return {"places": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
