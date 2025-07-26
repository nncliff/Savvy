import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const { query } = await req.json();
  // Use deepre GraphRAG service
  const graphragRes = await fetch("http://deepre:8000/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ 
      query,
      method: "global",
      root: "/app/user_storage/zhan.chen_gmail.com"
    }),
  });
  const data = await graphragRes.json();
  
  // Transform the response to match the expected chat format
  const response = {
    response: data.result || data.detail || "No response received",
    query: data.query || query,
    method: data.method || "global"
  };
  
  return NextResponse.json(response, { status: graphragRes.status });
}
