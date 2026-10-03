# Registers the daily job at 15:45 New York time on US trading weekdays, for the current user.
# If the PC was off at that time the job runs at next start; Yahoo still serves the closing quotes then.
# Remove with: Unregister-ScheduledTask -TaskName "SpotVol daily snapshot" -Confirm:$false
$name = "SpotVol daily snapshot"
$et = [TimeZoneInfo]::FindSystemTimeZoneById("Eastern Standard Time")
$nyNow = [TimeZoneInfo]::ConvertTime([datetime]::Now, $et)
$ny = [datetime]::SpecifyKind($nyNow.Date.AddHours(15).AddMinutes(45), [DateTimeKind]::Unspecified)
$local = [TimeZoneInfo]::ConvertTime($ny, $et, [TimeZoneInfo]::Local)
$days = "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"
if ($local.Date -gt $ny.Date) { $days = "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" }

$script = Join-Path $PSScriptRoot "daily.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At $local.ToString("HH:mm")
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)
Register-ScheduledTask -TaskName $name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
"Registered '$name' at $($local.ToString('HH:mm')) local time on $($days -join ', ')"
