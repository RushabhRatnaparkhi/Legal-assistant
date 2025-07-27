import mongoose from "mongoose";

const MessageSchema = new mongoose.Schema({
  chatId: { type: mongoose.Schema.Types.ObjectId, ref: "Chat", required: true },
  sender: { type: String, enum: ["user", "ai"], required: true },
  content: { type: String, required: true },
  createdAt: { type: Date, default: Date.now }
});

export default mongoose.models.Message || mongoose.model("Message", MessageSchema);