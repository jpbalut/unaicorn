def fake_predict(user_age):
	if int(user_age)>10:
		prediction = "survive"
	else:
		prediction = "super survive"
	return prediction