
import sqlite3
def main():
    dbFile = "rooms.sqlite"
    rooms = []
    for i in range(5):
        rooms.append("room" + str(i))
    print(rooms)
    con = sqlite3.connect(dbFile)
    cur = con.cursor()
    for room in rooms: #create tables
        cur.execute("CREATE TABLE " + room + "(day, time, day_in_week, class_in_session, is_peak_hours, is_240, is_500, 1h_prior_count, count, category)")
    res = cur.execute("select name from sqlite_master")
    print(res.fetchall())
if __name__ == "__main__":
    main()