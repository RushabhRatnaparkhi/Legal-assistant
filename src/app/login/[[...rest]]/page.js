'use client'
import { ClerkProvider, SignedIn, SignedOut, SignIn } from "@clerk/nextjs";

export default function LoginPage() {
  return (
    <ClerkProvider>
      <main className="min-h-screen flex items-center justify-center bg-gray-100">
        <SignedOut>
          <div className="bg-white shadow-xl rounded-2xl p-8 w-full max-w-md">
            <h1 className="text-2xl font-bold mb-6 text-center text-gray-800">Login to AI Legal Assistant</h1>
            <SignIn />
          </div>
        </SignedOut>
        <SignedIn>
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-4">You are already signed in!</h2>
            <a href="/" className="text-blue-600 underline">Go to Home</a>
          </div>
        </SignedIn>
      </main>
    </ClerkProvider>
)}
