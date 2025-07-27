import { connectDB } from "@/app/db";
import { NextResponse } from "next/server";
import Message from "@/app/models/Message";

export async function GET(request) {
  await connectDB();
  const { searchParams } = new URL(request.url);
  const chatId = searchParams.get("chatId");
  const messages = await Message.find({ chatId }).sort({ createdAt: 1 });
  return NextResponse.json(messages);
}

export async function POST(request) {
  await connectDB();
  const body = await request.json();
  const message = await Message.create(body);
  return NextResponse.json(message);
}