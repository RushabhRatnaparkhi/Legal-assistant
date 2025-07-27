'use client';
import { useRouter } from "next/navigation";
import { useClerk } from "@clerk/nextjs";

export default function Navbar() {
  const router = useRouter();
  const { signOut } = useClerk();

  return (
    <nav className="w-full bg-white shadow-md px-6 py-4 flex items-center justify-between">
      <div className="flex items-center space-x-3">
        <span className="text-xl font-bold text-blue-700">AI Legal Assistant</span>
      </div>
      <div className="flex items-center space-x-4">
        <a href="/" className="text-gray-700 hover:text-blue-600 font-medium">Home</a>
        <a href="/about" className="text-gray-700 hover:text-blue-600 font-medium">About</a>
        <button
          className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 font-semibold"
          onClick={() => signOut(() => router.replace("/login"))}
        >
          Logout
        </button>
      </div>
    </nav>);}