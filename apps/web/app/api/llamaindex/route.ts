import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const { query } = await req.json();
  // Use Docker internal hostname for llamaindex
  const llamaRes = await fetch("http://llamaindex:8080/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  const data = await llamaRes.json();
  return NextResponse.json(data, { status: llamaRes.status });
}
