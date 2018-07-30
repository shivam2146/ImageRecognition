from django import forms


class Digit(forms.Form):
	pic = forms.FileField(
		label='Select a file',
        widget=forms.FileInput(attrs={'class': 'custom-file-input', 'id': 'validatedCustomFile'}),
        label_suffix = ':-'
    	)