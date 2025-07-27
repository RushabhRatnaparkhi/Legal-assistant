import { NextResponse } from "next/server";

export async function POST(request) {
    const { prompt, contractText } = await request.json();

    // Call Python backend
    const response = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ question: prompt, contract: contractText }),
    });

    const data = await response.json();

    return NextResponse.json({ response: data.response });
}
