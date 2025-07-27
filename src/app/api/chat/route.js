import { connectDB } from "@/app/db";
import { NextResponse } from "next/server";
import Chat from "@/app/models/Chat"; 

export async function GET(request) {
  const response1 = await connectDB();
  return NextResponse.json({ response1 });
}

export async function POST(request) {
  await connectDB();
  const body = await request.json();
  const chat = await Chat.create(body); // Save chat
  return NextResponse.json(chat);
}