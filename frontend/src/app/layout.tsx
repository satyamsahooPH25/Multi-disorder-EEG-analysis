import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "CognitiveScreen — Multi-Disorder EEG Analysis",
  description:
    "HAMD-Net hybrid attention model for EEG-based detection of Alzheimer's, Parkinson's, FTD, and Schizophrenia.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen flex">
        <Sidebar />
        <main className="flex-1 ml-60 p-8">
          <div className="max-w-[1400px] mx-auto">{children}</div>
        </main>
      </body>
    </html>
  );
}
