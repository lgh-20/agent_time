# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from test import agent_executor          # 直接复用你现成的 agent
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ChatGLM-4 报时机器人")

# 允许前端跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Msg(BaseModel):
    user_input: str

@app.post("/chat")
def chat(msg: Msg):
    """
    每次请求只带用户当前输入，
    历史由 AgentExecutor 内部的 memory 自动维护。
    """
    try:
        result = agent_executor.invoke({"input": msg.user_input})
        return {"response": result["output"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)