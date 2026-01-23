#!/usr/bin/env python3
"""
Code Quality Validation Agent
Analyzes code changes for quality issues and improvement suggestions
"""

import os
import sys
import anthropic
import subprocess
import json
from pathlib import Path

def get_changed_files():
    """Get list of changed files"""
    try:
        with open('changed_files.txt', 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        return files
    except FileNotFoundError:
        return []

def read_file_content(file_path):
    """Read full file content"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

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

def analyze_code_quality(file_path, content, diff):
    """Use Claude to analyze code quality and suggest improvements"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return "Error: ANTHROPIC_API_KEY not set"

    client = anthropic.Anthropic(api_key=api_key)

    # Determine file type
    ext = Path(file_path).suffix
    lang_map = {
        '.c': 'C',
        '.h': 'C header',
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.cpp': 'C++',
        '.java': 'Java',
        '.go': 'Go'
    }
    language = lang_map.get(ext, 'code')

    prompt = f"""You are a senior code reviewer focusing on code quality and best practices.

Analyze this {language} code change and provide:

1. **Code Quality Issues** (0-10 score):
   - Readability
   - Maintainability
   - Performance
   - Error handling

2. **Specific Improvements**:
   - What should be changed and why
   - Concrete code suggestions

3. **Best Practices**:
   - Design patterns that could be applied
   - Industry standards being violated

4. **Refactoring Opportunities**:
   - Code duplication
   - Complex functions that should be split
   - Magic numbers or hardcoded values

5. **Security Concerns** (if applicable)

File: {file_path}

Recent Changes:
```diff
{diff}
```

Full File Context:
```{ext[1:]}
{content}
```

Format your response as:

## Quality Score: X/10

### Critical Issues
- [List critical problems]

### Improvements
- [Specific, actionable suggestions]

### Refactoring Ideas
- [Long-term improvement suggestions]

### Security Notes
- [Any security concerns]

Be constructive and specific. Prioritize actionable feedback."""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error calling Claude API: {e}"

def post_comment_to_pr(analysis_results):
    """Post quality analysis as PR comment"""
    pr_number = os.getenv('PR_NUMBER')
    if not pr_number:
        print("Not a PR event, skipping comment")
        return

    try:
        subprocess.run(
            ['gh', 'pr', 'comment', pr_number, '--body', analysis_results],
            check=True
        )
        print(f"✅ Posted quality analysis to PR #{pr_number}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error posting comment: {e}")

def generate_summary(all_analyses):
    """Generate overall quality summary"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return "Summary unavailable"

    client = anthropic.Anthropic(api_key=api_key)

    prompt = f"""Based on these individual file analyses, provide a brief overall assessment:

{all_analyses}

Provide:
1. Overall quality trend (improving/declining/stable)
2. Top 3 priority improvements
3. Positive highlights

Keep it concise (3-4 sentences)."""

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except:
        return "Summary generation failed"

def main():
    print("🔍 Code Quality Validation Agent Starting...\n")

    changed_files = get_changed_files()
    if not changed_files:
        print("✅ No files to analyze")
        return

    print(f"📋 Analyzing {len(changed_files)} file(s)\n")

    all_analyses = []
    results_by_file = {}

    for file_path in changed_files:
        # Skip non-code files
        code_extensions = {'.c', '.h', '.py', '.js', '.ts', '.cpp', '.java', '.go', '.rs'}
        if not any(file_path.endswith(ext) for ext in code_extensions):
            print(f"⏭️  Skipping {file_path} (not a code file)")
            continue

        print(f"🔎 Analyzing: {file_path}")

        # Read full file
        content = read_file_content(file_path)
        if content.startswith("Error"):
            print(f"  ⚠️  {content}")
            continue

        # Get diff
        diff = get_file_diff(file_path)
        if not diff or diff.startswith("Error"):
            print(f"  ⚠️  No diff available")
            continue

        # Analyze quality
        analysis = analyze_code_quality(file_path, content, diff)

        print(f"\n{'='*60}")
        print(f"Quality Analysis: {file_path}")
        print(f"{'='*60}")
        print(analysis)
        print(f"{'='*60}\n")

        all_analyses.append(f"### 📁 {file_path}\n\n{analysis}\n")
        results_by_file[file_path] = analysis

    if all_analyses:
        # Generate summary
        summary = generate_summary('\n\n'.join(all_analyses))

        # Create full report
        full_report = f"""# 🔍 Code Quality Analysis Report

{summary}

---

{''.join(all_analyses)}

---

*Automated quality check by Claude Code Quality Agent*
"""

        post_comment_to_pr(full_report)

        # Save results to file for artifact
        with open('quality_report.json', 'w') as f:
            json.dump({
                'summary': summary,
                'files': results_by_file
            }, f, indent=2)

        print("\n✅ Quality analysis complete!")
        print(f"📄 Results saved to quality_report.json")

    else:
        print("ℹ️  No code files to analyze")

if __name__ == "__main__":
    main()
