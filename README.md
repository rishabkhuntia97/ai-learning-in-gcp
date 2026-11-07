# AI Learning in GCP

My daily log of learning AI on Google Cloud. Each day gets its own folder with the notebooks I ran and short notes on what I learned.

## Structure

| Folder | What goes there |
|---|---|
| `days/YYYY-MM-DD-topic/` | One folder per day: notebooks plus `notes.md` |
| `projects/` | Bigger work that spans several days |
| `utils/` | Helper code reused across days |
| `scripts/` | `start_day.sh` and `end_day.sh` |
| `data/` | Local data only, never committed (kept in Cloud Storage instead) |

## Daily workflow

**Start of day** (on a fresh instance):
```bash
git clone https://github.com/<your-username>/ai-learning-in-gcp.git && cd ai-learning-in-gcp && bash scripts/start_day.sh "topic-name"
```
Create today's notebooks inside the folder it prints.

**End of day** (save notebooks first, then):
```bash
bash scripts/end_day.sh
```
Check that today's folder is on GitHub, then delete the instance.

## Learning log

| Date | Topic | Link |
|---|---|---|
