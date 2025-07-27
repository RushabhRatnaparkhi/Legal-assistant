import mongoose from "mongoose";

const ChatSchema = new mongoose.Schema({
  userId: { type: String, required: true },
  title: { type: String },
  createdAt: { type: Date, default: Date.now }
});

export default mongoose.models.Chat || mongoose.model("Chat", ChatSchema);