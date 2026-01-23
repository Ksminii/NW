#!/usr/bin/env python3
import os
import sys
import anthropic
import subprocess

def get_changed_files():
    """Get list of changed files from the workflow"""
    try:
        with open('changed_files.txt', 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        return files
    except FileNotFoundError:
        return []

def get_file_diff(file_path):
    """Get the diff for a specific file"""
    try:
        if os.getenv('GITHUB_EVENT_NAME') == 'pull_request':
            base_sha = os.getenv('GITHUB_BASE_REF', 'main')
            result = subprocess.run(
                ['git', 'diff', f'origin/{base_sha}', '--', file_path],
                capture_output=True,
                text=True
            )
        else:
            result = subprocess.run(
                ['git', 'diff', 'HEAD~1', 'HEAD', '--', file_path],
                capture_output=True,
                text=True
            )
        return result.stdout
    except Exception as e:
        return f"Error getting diff: {e}"

def review_code_with_claude(file_path, diff_content):
    """Send code to Claude for review"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set"

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Please review the following code changes in {file_path}.
Provide:
1. Brief summary of changes
2. Potential issues or bugs
3. Performance concerns
4. Code quality suggestions
5. Security considerations (if applicable)

Diff:
```
{diff_content}
```

Keep the review concise and actionable."""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error calling Claude API: {e}"

def post_comment_to_pr(review_text):
    """Post review as PR comment using GitHub CLI"""
    pr_number = os.getenv('PR_NUMBER')
    if not pr_number:
        print("Not a PR event, skipping comment")
        return

    try:
        subprocess.run(
            ['gh', 'pr', 'comment', pr_number, '--body', review_text],
            check=True
        )
        print(f"Posted review comment to PR #{pr_number}")
    except subprocess.CalledProcessError as e:
        print(f"Error posting comment: {e}")

def main():
    print("🤖 Starting Claude Code Review\n")

    changed_files = get_changed_files()
    if not changed_files:
        print("No files changed")
        return

    print(f"Reviewing {len(changed_files)} file(s):\n")

    all_reviews = []

    for file_path in changed_files:
        # Skip non-code files
        if not file_path.endswith(('.c', '.h', '.py', '.js', '.ts', '.java', '.cpp', '.go')):
            print(f"⏭️  Skipping {file_path} (not a code file)")
            continue

        print(f"📝 Reviewing: {file_path}")

        diff = get_file_diff(file_path)
        if not diff or diff.startswith("Error"):
            print(f"  ⚠️  Could not get diff for {file_path}")
            continue

        review = review_code_with_claude(file_path, diff)

        print(f"\n{'='*60}")
        print(f"Review for: {file_path}")
        print(f"{'='*60}")
        print(review)
        print(f"{'='*60}\n")

        all_reviews.append(f"## 📁 {file_path}\n\n{review}\n")

    if all_reviews:
        full_review = "# 🤖 Claude Code Review\n\n" + "\n---\n\n".join(all_reviews)
        full_review += "\n\n*Automated review by Claude Code Action*"
        post_comment_to_pr(full_review)

    print("✅ Review complete!")

if __name__ == "__main__":
    main()
