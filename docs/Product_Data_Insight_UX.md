# Product Definition: Data Insight UX Strategy
**"Invisible Intelligence" for the Minimalist Morning**

> **Core Philosophy**:  
> The system should become *smarter*, not *louder*.  
> Insights must be "Don't Make Me Think" moments—glanceable, non-intrusive, and appearing *only* when they add value.  
> We reject the "Dashboard" paradigm. We adopt the "Whisper" paradigm.

---

## 1. Design Principles (The "How")

1.  **Zero-blocker Input**: The "Morning Goal" < 15s rule is sacred. Insights must *never* interrupt the flow of writing the goal.
    *   *Decision*: No real-time analysis interference. No "Suggester" popup while typing.
2.  **Post-Action Reward**: Insights are the "reward" for the habit loop, appearing *immediately after* the "Commit" action.
    *   *Metaphor*: "Fortune Cookie" or "Daily Horoscope" — A small piece of wisdom unlocked by your action.
3.  **Contextual Integration**: Insights live *in the stream*, not on a separate page.
4.  **One Thing at a Time**: Never overwhelm. Show *one* dominant insight per day.

---

## 2. Interaction Flows

### Flow A: The Daily Input (Standard Morning)

1.  **User Opens App**: Clean input field (as is).
2.  **User Types Goal**: "Finishing the Q3 Report."
3.  **User Long-presses to Commit**: The satisfaction animation plays.
4.  **Transition**: The goal moves to the "History" list (or stays as "Today's Card").
5.  **The "Whisper" (New)**: 
    *   Immediately below Today's Goal card, a small, elegant **Insight Pill** or **Insight Card** fades in.
    *   *Example*: "Keep it up! 🎯 This is your 3rd Work goal this week. You're on a roll."
    *   *Action*: User glances, smiles (or ignores), and closes app. 
    *   *Friction Added*: 0 seconds.

### Flow B: "History on this Day" (Nostalgia)

*   **Trigger**: User opens app, and it is a significant date (e.g., 1 year since first use, or exact date match).
*   **Presentation**: 
    *   *Before Input*: A translucent card floats above the input: "On this day last year, you focused on 'Learning Swift'. What's today's focus?"
    *   *Why Before?*: To inspire today's goal (priming).
    *   *Dispensability*: Tap anywhere to dismiss instantly.

### Flow C: The Weekly Review (Passive)

*   **Trigger**: Sunday Morning (or user's preferred "Review Day").
*   **Presentation**: 
    *   Upon committing Sunday's goal, the "Reward" is a slightly larger card: **"Weekly Briefing"**.
    *   Shows: Topic distribution visual (minimalist bar/pie) + One text summary.
    *   *Text*: "You balanced Work (3) and Health (2) well this week."

---

## 3. Insight Logic & Presentation Levels

We categorize insights by **Intrusiveness** and **Priority**.

### Level 1: Ambient (Micro-Feedback)
*   *Where*: Small icon or color tint on the Goal Card history.
*   *Content*: Topic Icon (Briefcase for Work, Heart for Health) + Sentiment color (Subtle warm/cool glow).
*   *Goal*: User scrolls history and *feels* the pattern without reading text.

### Level 2: The "Whisper" (Daily Post-Commit)
*   *Where*: Single line text appearing under today's committed goal.
*   *Content Source*: **PatternAnalyzer** or **TrendAnalyzer**.
*   *Examples*:
    *   *Pattern*: "Monday is usually your 'Deep Work' day."
    *   *Trend*: "You've been consistently Positive for 5 days!"
    *   *Detail*: "This goal is highly specific. Great job." (SpecificityClassifier)

### Level 3: The "Spotlight" (High Priority / Rare)
*   *Where*: A dedicated card that inserts itself into the top of the History Stream.
*   *Content Source*: **BalanceAdvisor** or "Milestones".
*   *Examples*:
    *   *Balance Warning*: "⚠️ 80% Work goals lately. Don't forget to rest."
    *   *Milestone*: "🏅 100th Goal Recorded!"
    *   *Nostalgia*: "📅 One year ago today..."

---

## 4. UI/UX Concept: "The Insight Stack"

Imagine the main interface as a vertical stack:

1.  **Header**: Date / Greeting.
2.  **Committed Goal Card**: The hero element.
3.  **Insight Container (Dynamic)**:
    *   *Empty* (Default state).
    *   *The Whisper*: A slim, rounded rectangle fitting the width of the goal card. Background: 5% opacity dynamic color.
    *   *The Card*: A full card for "Weekly Review" or "Nostalgia".
4.  **History Stream**: The past goals fading out below.

### Visual Style
*   **Typography**: System Serif for the Goal (Personal), System Sans-serif (Rounded) for the Insight (AI/System voice).
*   **Color**: 
    *   Goals: High contrast black/white.
    *   Insights: Muted, pastel colors. "Information, not navigation."

---

## 5. Technical Mappings

| UI Component | Data Source (Backend) | Trigger |
| :--- | :--- | :--- |
| **Topic Icon** | `TopicClassifier` | On Commit |
| **The Whisper** | `InsightGenerator` (Top-1 priority) | On Commit (Async calc) |
| **Memory Card** | `CoreData` (Date Query) | On App Open (Pre-calc) |
| **Balance Warn** | `BalanceAdvisor` | Weekly Task / On Commit |

## 6. Next Steps for Implementation

1.  **Refine "InsightGenerator"**: Ensure it outputs a single "Best" insight string for the `Whisper` UI.
2.  **Design the "Insight Pill"**: Build a SwiftUI view component that can render `InsightType` visually.
3.  **Modify `ContentView`**: Insert the "Insight Container" into the post-commit animation sequence.
